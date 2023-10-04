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

    def forward(self, x, pad_mask):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        attn = self.c_attn(x)
        q, k, v = attn.split(self.n_embd, dim=2)
        # to-do: modify q, k and v such that they are of shape (B, nh, T, hs)

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
        y = self.c_proj(y)
        y = self.resid_dropout(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config["n_embd"], 4 * config["n_embd"])
        self.gelu = nn.GELU()
        # to-do: specify appropriate in_features and out_features for the linear projection
        self.c_proj = nn.Linear(, )
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config["n_embd"])
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config["n_embd"])
        self.mlp = MLP(config)

    def forward(self, x, pad_mask):
        x = x + self.attn(self.ln_1(x), pad_mask)
        # to-do: forward pass through the mlp
        
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config["vocab_size"], config["n_embd"], padding_idx=config.get("pad_token", None)),
            wpe=nn.Embedding(config["block_size"], config["n_embd"]),
            drop=nn.Dropout(config["dropout"]),
            h=nn.ModuleList([Block(config) for _ in range(config["n_layer"])]),
            ln_f=nn.LayerNorm(config["n_embd"])))
        self.lm_head = nn.Linear(config["n_embd"], config["vocab_size"], bias=False)
        self.transformer.wte.weight = (self.lm_head.weight)  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config["n_layer"]))

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
        assert t <= self.config["block_size"], f"Cannot forward len {t}, only {self.config['block_size']}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        
        # to-do: add token and position embeddings. Then apply dropout "transformer.drop". Receive the output in "x"

        for block in self.transformer.h:
            x = block(x, pad_mask)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            # to-do: apply cross entropy loss. Also set ignore index to be the "pad_token" from config
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
            idx_cond = idx if idx.size(1) <= self.config["block_size"] else idx[:, -self.config["block_size"] :]
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
