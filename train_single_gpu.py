import os
import sys
with open(sys.argv[0]) as f:
    filecode = f.read() # read the code of this file ASAP, for logging
import yaml
import math
import time
import random
import inspect
import argparse
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

import torch
import tiktoken
import torch.nn as nn
import torch.nn.functional as F

from model import GPT, GPTConfig
from utils import save_checkpoint
from dataloader import FineWebEduDataLoader
from hellaswag import render_example, iterate_examples, get_most_likely_row

run_id = datetime.now().strftime("%Y%m%d")
print(f'run id: {run_id}')

# -----------------------------------------------------------------------------
# model loss estimation and sampling code
def calc_loss_loader(data_loader, model, device, num_batches, print_loss=True) -> float:
    model.eval()
    data_loader.reset()

    with torch.no_grad():
        loss_accum = 0.0
        loss_steps = num_batches
        for _ in range(loss_steps):
            x, y = data_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, (loss, aux_loss, router_z_loss) = model(x, y)
            loss = loss / loss_steps
            loss_accum += loss.detach()
    
    if print_loss:
        # averaged per-step loss, averaged over `num_batches` batches or steps
        print(f"Validation loss: {loss_accum.item():.4f}")
    
    model.train()
    return loss_accum.item()


# caution: this is not distributed, so it will only work on a single gpu
def calc_hella_accuracy(model, device, print_acc=True) -> float:
    num_correct_norm = 0
    num_total = 0

    for i, example in enumerate(iterate_examples("val")):
        _, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)
        # get the logits
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, (loss, aux_loss, router_z_loss) = model(tokens)
            pred_norm = get_most_likely_row(tokens, mask, logits)
        num_total += 1
        num_correct_norm += int(pred_norm == label)
    
    acc_norm = num_correct_norm / num_total
    if print_acc:
        print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
    return acc_norm


# caution: this is not distributed, so it will only work on a single gpu
def generate_and_print_samples(model, tokenizer, device,
                               num_return_sequences = 4,
                               max_length = 32,
                               start_context = "Hello, I'm a language model,",
                               random_seed = 42
                               ):
    
    model.eval()
    encoder = None
    decoder = None
    if hasattr(tokenizer, 'encode'):
        encoder = tokenizer.encode
        decoder = tokenizer.decode
    elif hasattr(tokenizer, 'tokenize'):
        encoder = tokenizer.tokenize
        decoder = tokenizer.detokenize
    else:
        raise ValueError(f"Please pass a tokenizer with either encode/decode or tokenize/detokenize methods")

    
    tokens = encoder(start_context)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    xgen = tokens.to(device)
    # don't interfere with other seeds
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(random_seed)

    while xgen.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(xgen) # (B, T, vocab_size)
            # take the logits at the last position
            # print(logits.shape, logits.dtype)
            logits = logits[:, -1, :] # (B, vocab_size)
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            # note: multinomial does not demand the input to sum to 1
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
            # append to the sequence
            xgen = torch.cat((xgen, xcol), dim=1)
    # print the generated text
    for i in range(num_return_sequences):
        tokens = xgen[i, :max_length].tolist()
        decoded = decoder(tokens)
        print(f"sample {i}: {decoded}")
    
    model.train()

# -----------------------------------------------------------------------------

def get_lr(it):
    # linear warmup
    if it < args.warmup_steps:
        return args.max_lr * (it + 1) / args.warmup_steps
    # if it > lr decay iters, return min_lr
    if it > args.max_steps:
        return args.min_lr
    decay_ratio = (it - args.warmup_steps) / (args.max_steps - args.warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # starts at 1, goes to 0
    return args.min_lr + coeff * (args.max_lr - args.min_lr)

# -----------------------------------------------------------------------------


@dataclass
class Args:
    # config
    # model_size = '150M'
    # sparse_value_frac = None # None for vanilla
    
    # details about the run
    name = 'moe'

    # data
    data_root='./edu_fineweb10B'

    # batch size and gradient accumulation
    total_batch_size = 524_288 # 2**19, closest power of 2 to ~0.5M
    B = 16    # 8 fits for 900M, 16 fits for 450M, 32 fits in one A100 40GB for 150M model
    T = 1024
    vocab_size = 50304
    use_compile = True
    
    # optimization
    max_lr = 6e-4 # prev constant lr that we were using
    min_lr = max_lr * 0.1
    warmup_steps = 2000 # to be consistent with the tokenformer paper
    max_steps =  19_073 * 10 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens

    # iters to estimate val loss
    val_loss_steps = 20

    # evaluation and logging
    val_loss_every = 250 # every how many steps to evaluate val loss? 0 for only at the end
    sample_from_model_every = 250
    save_checkpoint_every = 10_000 # 0 for no checkpoints
    logs_dir = './logs'
    checkpoint_dir = './checkpoints'
    save_checkpoint = False
    
    # moe related args
    n_exp: int = 8 # if n_exp = 1 we just use regular MLP layers
    top_k: int = 2
    
    use_aux_loss: bool = False      # apply auxiliary loss (from Switch Transformer) in router
    use_router_z_loss: bool = False # apply router z loss (from ST-MoE)
    use_noisy_top_k: bool = False    # use noisy router
    
    # loss weighting: how did they decide these values?
    aux_loss_weight: float = 0.01 # default setting from Switch Transformer (see top of page 8)
    router_z_loss_weight: float = 0.001 # default setting from ST-MoE (see page 8 eq. 6)
    
    # it is the capacity factor
    train_capacity: float = 1.25  # default setting from ST-MoE (see top of page 6)
    eval_capacity: float = 2.0
    
    # capacity: minimum batch size to send to any single expert
    min_capacity: int = 4  # minimum batch size to send to any single expert
    
    # how often to convert a layer to an MoE layer
    stride: int = 2 # one in every stride layers are converted to an MoE
    
    # weight init scheme from Switch Transformer
    use_switch_tfm_init: bool = False  # use weight init scheme from Switch Transformer
    switch_tfm_init_scale: float = 1.0
    
    # this is to be used in router forward and in a context manager
    router_use_full_prec: bool = False  # use float32 precision in the router

args = Args()

# Parse command line arguments to override defaults
parser = argparse.ArgumentParser(description='Train GPT-2 MoE model')
parser.add_argument('--B', type=int, default=args.B,
                    help='Batch size')
parser.add_argument('--T', type=int, default=args.T,
                    help='Sequence length')
parser.add_argument('--seed', type=int, default=1337,
                    help='Random seed')
parser.add_argument('--max_lr', type=float, default=args.max_lr,
                    help='Maximum learning rate')
parser.add_argument('--max_steps', type=int, default=args.max_steps,
                    help='Maximum training steps')
parser.add_argument('--val_loss_every', type=int, default=args.val_loss_every,
                    help='Evaluate validation loss every N steps')
parser.add_argument('--save_checkpoint_every', type=int, default=args.save_checkpoint_every,
                    help='Save checkpoint every N steps')
parser.add_argument('--total_batch_size', type=int, default=args.total_batch_size,
                    help='Total batch size in tokens')
parser.add_argument('--data_root', type=str, default=args.data_root,
                    help='Path to training data')
parser.add_argument('--use_compile', action='store_true', default=args.use_compile,
                    help='Use torch.compile')
parser.add_argument('--no_compile', dest='use_compile', action='store_false',
                    help='Disable torch.compile')
# moe related args
parser.add_argument('--n_exp', type=int, default=args.n_exp,
                    help='Number of experts')
parser.add_argument('--top_k', type=int, default=args.top_k,
                    help='Top k for noisy top k')
parser.add_argument('--use_noisy_top_k', action='store_true', default=args.use_noisy_top_k,
                    help='Use noisy top k')
parser.add_argument('--use_aux_loss', action='store_true', default=args.use_aux_loss,
                    help='Use auxiliary loss')
parser.add_argument('--use_router_z_loss', action='store_true', default=args.use_router_z_loss,
                    help='Use router z loss')
parser.add_argument('--use_switch_tfm_init', action='store_true', default=args.use_switch_tfm_init,
                    help='Use switch transformer initialization')
parser.add_argument('--router_use_full_prec', action='store_true', default=args.router_use_full_prec,
                    help='Use full precision in the router')



cmd_args = parser.parse_args()

# Override args with command line arguments
args.B = cmd_args.B
args.T = cmd_args.T
args.max_lr = cmd_args.max_lr
args.max_steps = cmd_args.max_steps
args.val_loss_every = cmd_args.val_loss_every
args.save_checkpoint_every = cmd_args.save_checkpoint_every
args.use_compile = cmd_args.use_compile
args.total_batch_size = cmd_args.total_batch_size
args.data_root = cmd_args.data_root

# moe related args
args.n_exp = cmd_args.n_exp
args.top_k = cmd_args.top_k
args.use_noisy_top_k = cmd_args.use_noisy_top_k
args.use_aux_loss = cmd_args.use_aux_loss
args.use_router_z_loss = cmd_args.use_router_z_loss
args.use_switch_tfm_init = cmd_args.use_switch_tfm_init
args.router_use_full_prec = cmd_args.router_use_full_prec

# fetch seed
seed = cmd_args.seed

# modify name:
if args.n_exp == 1:
    args.name = "no_moe"
    
if args.top_k == 1: # use only 1 expert to see performance improvement if any
    args.name += "_top_k_1"

if args.use_noisy_top_k:
    args.name = args.name + "_noisy_top_k"
if args.use_aux_loss:
    args.name = args.name + "_aux_loss"
if args.use_router_z_loss:
    args.name = args.name + "_router_z_loss"
if args.use_switch_tfm_init:
    args.name = args.name + "_switch_tfm_init"
if args.router_use_full_prec:
    args.name = args.name + "_router_full_prec"
    
print(f"Training model: {args.name}")

# -----------------------------------------------------------------------------
# dataloader and tokenizer
enc = tiktoken.get_encoding("gpt2")

assert args.total_batch_size % (args.B * args.T) == 0, "total batch size in number of tokens should be divisible by B*T"
grad_accum_steps = args.total_batch_size // (args.B * args.T)
print(f"total desired batch size: {args.total_batch_size} tokens")
print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

# initialize the dataloader
train_loader = FineWebEduDataLoader(B=args.B, T=args.T, split='train')
val_loader = FineWebEduDataLoader(B=args.B, T=args.T, split='val')


# device init
# enable tf32, now matmuls will use tf32 (tensor cores from A100)
torch.set_float32_matmul_precision('high') # default is highest

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.mps.is_available():
    device = "mps"
print(f"using device: {device}")

random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
# -----------------------------------------------------------------------------
config = GPTConfig(vocab_size=50304,
                   n_exp=args.n_exp,
                   top_k=args.top_k,
                   use_noisy_top_k=args.use_noisy_top_k,
                   use_aux_loss=args.use_aux_loss,
                   use_router_z_loss=args.use_router_z_loss,
                   use_switch_tfm_init=args.use_switch_tfm_init,
                   router_use_full_prec=args.router_use_full_prec)
model = GPT(config)
model.to(device)

if args.use_compile:
    model = torch.compile(model)

optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device) # fused update
# -----------------------------------------------------------------------------

log_dir = args.logs_dir
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log_{args.name}_{run_id}_seed_{seed}.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

# begin by printing the code in this file
# print(filecode)

checkpoint_dir = os.path.join(args.checkpoint_dir, f"{args.name}_{run_id}_seed_{seed}") # folderize checkpoints for this run
os.makedirs(checkpoint_dir, exist_ok=True)

for step in range(args.max_steps):

    last_step = (step == args.max_steps - 1)

    # once in a while evaluate on the validation set
    if step % args.val_loss_every == 0 or last_step:
        val_loss_current = calc_loss_loader(val_loader, model, device, num_batches=args.val_loss_steps)
        with open(log_file, "a") as f:
            f.write(f"{step} val {val_loss_current:.4f}\n")

    # once in a while save checkpoint
    if (step > 0 and step % args.save_checkpoint_every == 0) or last_step:
        save_checkpoint(model, optimizer, name=f"{args.name}_seed_{seed}_step{step:06d}", root_dir=checkpoint_dir)

    # once in a while evaluate on hellaswag
    if step % args.val_loss_every == 0 or last_step:
        hacc = calc_hella_accuracy(model, device, print_acc=True)
        with open(log_file, "a") as f:
            f.write(f"{step} hella {hacc:.4f}\n")
    
    # once in a while sample from the model
    if step % args.sample_from_model_every == 0 or last_step:
        generate_and_print_samples(model, enc, device)

    # start timer
    t0 = time.time()
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    aux_loss_accum = 0.0
    router_z_loss_accum = 0.0

    # gradient-accumulation
    for micro_step in range(grad_accum_steps):
        # data loading
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
    
        # forward-backward and step
        # amp: just surround forward pass and loss calculation, only possible in A100
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, (loss, aux_loss, router_z_loss) = model(x, y)
        loss = loss / grad_accum_steps
        if aux_loss is not None:
            aux_loss = aux_loss / grad_accum_steps
        if router_z_loss is not None:
            router_z_loss = router_z_loss / grad_accum_steps
        
        # accumulate losses
        loss_accum += loss.detach()
        if aux_loss is not None:
            aux_loss_accum += aux_loss.detach()
        if router_z_loss is not None:
            router_z_loss_accum += router_z_loss.detach()
        
        loss.backward() # deposits gradients, i.e., += on nodes

    # clip gradient norms to 1.0, returns total norm of the gradient vector
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # determine lr for this step
    lr = get_lr(step)
    # pytorch syntax to set the learning rate for the parameters
    for param_group in optimizer.param_groups:
        # param_group is a dict
        param_group['lr'] = lr
    optimizer.step()

    # wait for gpu to finish the compute and measure time
    torch.cuda.synchronize()
    t1 = time.time()

    dt = (t1 - t0)*1000 # time difference for one-batch or step in miliseconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    tps = tokens_processed / (t1 - t0)

    memory_used = torch.cuda.max_memory_allocated() // 1024 // 1024 // 1024 if torch.cuda.is_available() else 0
    
    print(f"Step {step:4d} | loss: {loss_accum.item():.6f} | aux_loss: {aux_loss_accum:.6f} | router_z_loss: {router_z_loss_accum:.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f}ms | tok/sec: {tps:.2f} | mem: {memory_used} GB")
    
    with open(log_file, "a") as f:
        f.write(f"{step} train {loss_accum.item():.6f}\n")


# -----------------------------------------------------------------------------
# at the very end of the file
print(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
      f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB")
print(f"Val loss for trained model {args.name}: {calc_loss_loader(val_loader, model, device, num_batches=50)}")