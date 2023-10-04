from model_exercise1_solution import GPT

pre_training = True  # specifies if its pre-training or fine-tuning
from_scratch = True  # specifies if we resuming from a checkpoint or training from scratch
task = ""  # name of task in case of fine-tuning. used to understand type of task and read its dataset
prompt_vocab_size = 0  # prompt-tuning. Supports only one task
classification_task = "classification" in task

model_config = dict(n_layer=6, n_head=6, n_embd=150, block_size=25,
                    vocab_size=vocab_size, dropout=0.2, pad_token=stoi["*"])
model_extended_config = dict()

always_save_checkpoint = False  # if True, we update the checkpoint after each training iteration
eval_only = False  # if True, only evaluation and no training

MODEL_DIR = "models/"
IN_CHECKPOINT = ""  # checkpoint to load the model from
OUT_CHECKPOINT = "base.pt"  # checkpoint to write the model to

learning_rate = 1e-3  # maximum learning rate
max_iters = 50000  # number of gradient update iterations
lr_decay_iters = 50000  # number of iterations to decay the learning rate
min_lr = 1e-4  # minimum learning rate
weight_decay = 1e-2  # regularization
warmup_iters = 500  # number of iterations to linearly increase learning rate till maximum value
grad_clip = 0.0  # gradient clipping beyond this value
decay_lr = False  # to enable decay of learning rate. If False, learning rate same for all the iterations

batch_size = 64
gradient_accumulation_steps = 1  # to simulate large batch sizes

eval_interval = 250  # evaluation and writing of models every eval_interval iterations
do_log = True  # enable logging
log_interval = 100  # log metrics about the model training every log_interval iterations
eval_iters = 100  # number of batches to evaluate on during evaluation