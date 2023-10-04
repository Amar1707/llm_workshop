import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config["n_embd"] % config["n_head"] == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config["n_embd"], 3 * config["n_embd"])
        # output projection
        self.c_proj = nn.Linear(config["n_embd"], config["n_embd"])
        # regularization
        self.attn_dropout = nn.Dropout(config["dropout"])
        self.resid_dropout = nn.Dropout(config["dropout"])
        self.n_head = config["n_head"]
        self.n_embd = config["n_embd"]
        self.dropout = config["dropout"]

        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias",torch.tril(torch.ones(config["block_size"], config["block_size"])).view(
                1, 1, config["block_size"], config["block_size"]))
        
        if config.get("lora_rank", 0) > 0:
            self.c_attn.requires_grad_(False)
            self.c_proj.requires_grad_(False)
            self.delta_a_c_attn = nn.Linear(config["n_embd"], config["lora_rank"], bias=False)
            self.delta_b_c_attn = nn.Linear(config["lora_rank"], 3 * config["n_embd"], bias=False)
            self.delta_a_c_proj = nn.Linear(config["n_embd"], config["lora_rank"], bias=False)
            self.delta_b_c_proj = nn.Linear(config["lora_rank"], config["n_embd"], bias=False)
            self.lora_scale = config.get("lora_alpha", 1) / config["lora_rank"]

    def forward(self, x, pad_mask):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        if hasattr(self, "lora_scale"):
            attn = (x @ self.c_attn.weight.t() + 
             (x @ self.delta_a_c_attn.weight.t() @ self.delta_b_c_attn.weight.t()) * self.lora_scale + self.c_attn.bias)
                
        else:
            attn = self.c_attn(x)
        q, k, v = attn.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        if pad_mask is not None:
            att = att.masked_fill(pad_mask[:, :, :T, :T] == 0, float("-inf"))
            att = torch.nan_to_num(att)
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (y.transpose(1, 2).contiguous().view(B, T, C))  # re-assemble all head outputs side by side

        # output projection
        if hasattr(self, "lora_scale"):
            y = (y @ self.c_proj.weight.t() +
             (y @ self.delta_a_c_proj.weight.t() @ self.delta_b_c_proj.weight.t()) * self.lora_scale + self.c_proj.bias)
        else:
            y = self.c_proj(y)
        y = self.resid_dropout(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config["n_embd"], 4 * config["n_embd"])
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config["n_embd"], config["n_embd"])
        self.dropout = nn.Dropout(config["dropout"])
        if config.get("lora_rank", 0) > 0:
            self.c_fc.requires_grad_(False)
            self.c_proj.requires_grad_(False)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

    
class Adapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.down_proj = nn.Linear(config["n_embd"], config["adapter_size"])
        self.nl = nn.ReLU()
        self.up_proj = nn.Linear(config["adapter_size"], config["n_embd"])
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, x):
        x = self.down_proj(x)
        x = self.nl(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        return x

    
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config["n_embd"])
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config["n_embd"])
        self.mlp = MLP(config)
        if config.get("adapter_size", 0) > 0:
            self.ln_1.requires_grad_(False)
            self.attn.requires_grad_(False)
            self.ln_2.requires_grad_(False)
            self.mlp.requires_grad_(False)
            self.ln_3 = nn.LayerNorm(config["n_embd"])
            self.adapter = Adapter(config)
        if config.get("lora_rank", 0) > 0:
            self.ln_1.requires_grad_(False)
            self.ln_2.requires_grad_(False)

    def forward(self, x, pad_mask):
        x = x + self.attn(self.ln_1(x), pad_mask)
        x = x + self.mlp(self.ln_2(x))
        if hasattr(self, "adapter"):
            x = x + self.adapter(self.ln_3(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.get("prompt_vocab_size", 0) > 0:
            self.prompt_encoder = nn.Embedding(config["prompt_vocab_size"], config["n_embd"])
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config["vocab_size"], config["n_embd"], padding_idx=config.get("pad_token", None)),
            wpe=nn.Embedding(config["block_size"], config["n_embd"]),
            drop=nn.Dropout(config["dropout"]),
            h=nn.ModuleList([Block(config) for _ in range(config["n_layer"])]),
            ln_f=nn.LayerNorm(config["n_embd"])))
        
        if config.get("n_classes", 0) <= 0:
            self.lm_head = nn.Linear(config["n_embd"], config["vocab_size"], bias=False)
            self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying
        else:
            self.c_head = nn.Linear(config["n_embd"], config["n_classes"], bias=False)

        if config.get("freeze", False):
            self.transformer.requires_grad_(False)
            
        if config.get("adapter_size", 0) or config.get("lora_rank", 0) > 0:
            self.transformer.wte.requires_grad_(False)
            self.transformer.wpe.requires_grad_(False)
            
        if config.get("prompt_vocab_size", 0) > 0:
            self.transformer.requires_grad_(False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight") or pn.endswith("up_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config["n_layer"]))
            if ".delta_b_" in pn:
                torch.nn.init.zeros_(p)
                
        if config.get("prompt_vocab_size", 0) > 0:
            with torch.no_grad():
                ix = torch.randint(config["vocab_size"], (config["prompt_vocab_size"],))
                weights = []
                for i, index in enumerate(ix):
                    weights.append(self.transformer.wte.weight[index])
                weights = torch.stack(weights)
                self.prompt_encoder.weight.copy_(weights)

        # report number of parameters
        print("total parameters:",sum(p.numel() for p in self.parameters() if p.requires_grad))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, prompts=None, pad_mask=None):
        device = idx.device
        b, t = idx.size()
        if prompts is not None:
            t += prompts.size()[1]
        assert t <= self.config["block_size"], f"Cannot forward len {t}, only {self.config['block_size']}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        if prompts is not None:
            prompts_emb = self.prompt_encoder(prompts)
            tok_emb = torch.cat((prompts_emb, tok_emb), dim=1)

        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x, pad_mask)
        x = self.transformer.ln_f(x)
        
        if hasattr(self, "c_head"):
            logits = self.c_head(x[:, -1, :])
            if targets is not None:
                # if we are given some desired targets also calculate the loss
                loss = F.cross_entropy(logits, targets.view(-1))
            else:
                loss = None
        else:
            if targets is not None:
                # if we are given some desired targets also calculate the loss
                logits = self.lm_head(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                       ignore_index=self.config.get("pad_token", -100))
            else:
                # inference-time mini-optimization: only forward the lm_head on the very last position
                logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
                loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate):
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        return optimizer

    @torch.no_grad()
    def generate(self,idx,max_new_tokens,temperature=1.0,top_k=None,end_token=None,prompt=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (idx if idx.size(1) <= self.config["block_size"] - self.config.get("prompt_vocab_size", 0)
                        else idx[:, -(self.config["block_size"] - self.config.get("prompt_vocab_size", 0)) :])
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond, prompts=prompt)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            if idx_next == end_token:
                break

        return idx
    
    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config["block_size"]
        self.config["block_size"] = block_size
        learnable = self.transformer.wpe.weight.requires_grad
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        self.transformer.wpe.requires_grad_(learnable)
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    def extend_vocab(self, n_added_tokens, pad_token=None):
        if pad_token is not None:
            self.config["pad_token"] = pad_token
        with torch.no_grad():
            new_embeddings = nn.Embedding(self.config["vocab_size"] + n_added_tokens,self.config["n_embd"],
                padding_idx=self.config.get("pad_token", None))
            new_embeddings.to(self.transformer.wte.weight.device,dtype=self.transformer.wte.weight.dtype)
            self._init_weights(new_embeddings)
            new_embeddings.weight.data[: self.config["vocab_size"], :] = (self.transformer.wte.weight.data[:
                                                                          self.config["vocab_size"], :])
            learnable = self.transformer.wte.weight.requires_grad
            self.transformer.wte = new_embeddings
            self.transformer.wte.requires_grad_(learnable)

            if hasattr(self, "lm_head"):
                new_lm_head = nn.Linear(self.config["n_embd"],self.config["vocab_size"] + n_added_tokens,bias=False)
                new_lm_head = new_lm_head.to(self.lm_head.weight.device, dtype=self.lm_head.weight.dtype)
                self._init_weights(new_lm_head)
                new_lm_head.weight.data[: self.config["vocab_size"], :] = (self.lm_head.weight.data[: 
                                                                           self.config["vocab_size"], :])
                learnable = self.lm_head.weight.requires_grad
                self.lm_head = new_lm_head
                self.lm_head.requires_grad_(learnable)
                self.transformer.wte.weight = self.lm_head.weight

            self.config["vocab_size"] += n_added_tokens

    
    @classmethod
    def from_pretrained(cls, model_type, config={}):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config.update(config_args)
        config["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config["block_size"] = 1024  # always 1024 for GPT model checkpoints

        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]  # same, just the mask (buffer)
        transposed = ["attn.c_attn.weight","attn.c_proj.weight","mlp.c_fc.weight","mlp.c_proj.weight",]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        # assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            elif k in sd_keys:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape, f"shapes of {k} are not matching"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
            else:
                print(k, "not found")

        return model
