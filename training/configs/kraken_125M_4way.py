batch_size = 20
block_size = 1024
gradient_accumulation_steps = 24

# batch_size * block_size * gradient_accumulation_steps
# The total number of tokens is about 147B
max_iters = 300000
lr_decay_iters = 300000
always_save_checkpoint = True

n_heads_in_split = 3 # 12/3 heads = 4 ways
ffn_expansion=2
n_embd = 504
is_split = True
compile = True

# eval stuff
eval_interval = 500
eval_iters = 100
log_interval = 10
eval_only = False

# weight decay
weight_decay = 1e-1

learning_rate = 2.5e-4
min_lr = 2.5e-5

init_from = "scratch"
out_dir = "CKPT_DIRECTORY"
data_dir = "DATASET_DIRECTORY"