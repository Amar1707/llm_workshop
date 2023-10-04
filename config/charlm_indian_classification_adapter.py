from model_exercise3_solution import GPT

pre_training = False  # specifies if its pre-training or fine-tuning
from_scratch = True  # specifies if we resuming from a checkpoint or training from scratch
task = "indian_classification"  # name of task in case of fine-tuning. used to understand type of task and read its dataset
prompt_vocab_size = 0  # prompt-tuning. Supports only one task
classification_task = "classification" in task

model_extended_config = dict(n_classes=2, adapter_size=5)

always_save_checkpoint = False  # if True, we update the checkpoint after each training iteration
eval_only = False  # if True, only evaluation and no training

MODEL_DIR = "models/"
IN_CHECKPOINT = "base.pt"  # checkpoint to load the model from
OUT_CHECKPOINT = "indian_classification_adapter.pt"  # checkpoint to write the model to

learning_rate = 1e-3  # maximum learning rate
max_iters = 2000  # number of gradient update iterations
lr_decay_iters = 2000  # number of iterations to decay the learning rate
min_lr = 1e-4  # minimum learning rate
weight_decay = 1e-2  # regularization
warmup_iters = 200  # number of iterations to linearly increase learning rate till maximum value
grad_clip = 0.0  # gradient clipping beyond this value
decay_lr = False  # to enable decay of learning rate. If False, learning rate same for all the iterations

batch_size = 32
gradient_accumulation_steps = 1  # to simulate large batch sizes

eval_interval = 50  # evaluation and writing of models every eval_interval iterations
do_log = True  # enable logging
log_interval = 20  # log metrics about the model training every log_interval iterations
eval_iters = 30  # number of batches to evaluate on during evaluation