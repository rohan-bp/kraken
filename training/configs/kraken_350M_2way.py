batch_size = 12
block_size = 1024
gradient_accumulation_steps = 40

# The total number of tokens is about 147B
max_iters = 300000
lr_decay_iters = 300000

learning_rate = 1.5e-4
min_lr = 1.5e-5

always_save_checkpoint = True

n_layer = 24
n_heads_in_split = 8
n_head = 16
ffn_expansion=2
n_embd = 888
is_split = True
compile = True

eval_only = False

# eval stuff
eval_interval = 1000
eval_iters = 100
log_interval = 10

# weight decay
weight_decay = 1e-1

init_from = "scratch"
out_dir = "CKPT_DIRECTORY"
data_dir = "DATASET_DIRECTORY"
