# build-gpt2-moe

This repository houses some of my experiments to understand how to train small (~GPT2 scale) LLMs and their Mixture-of-Experts (MoEs) variants.

<p align="center">
  <img src="https://github.com/swag2198/build-gpt2/blob/main/results/figures/val_42_hella_42.png?raw=true" alt="val loss and hella for different MoE variants"/>
</p>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#dataset-used">Dataset used</a></li>
    <li><a href="#model-description">Repository structure</a></li>
    <li><a href="#hyperparameters">Hyperparameters</a></li>
    <li><a href="#compute-resources">Compute resources</a></li>
    <li><a href="#results">Results</a></li>
    <li><a href="#references-and-acknowledgements">References</a></li>
  </ol>
</details>


## Dataset used
All models are trained on the [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) dataset, particularly on the `sample-10BT` subset. To download the 
dataset and create the shards, just run `python fineweb.py`, it will create a folder `edu_fineweb10B/` in the current directory and store 99 training shards and 1
validation shard. Each shard contains 100M tokens.

For additional evaluation, I use the HellaSwag eval set, running `hellaswag.py` will download the data
and put it in another folder: `hellaswag/hellaswag_val.jsonl`.

## Model description
The model is defined in [model.py](https://github.com/swag2198/build-gpt2/blob/main/model.py). It is identical to the model used in [nanoMoE](https://github.com/wolfecameron/nanoMoE). What I understood from reading some papers and blog posts, is that there are quite a few problems that occur during MoE 
training (imbalanced expert selection, router logits get too high, mixed-precision error, MoE modules weight initialization etc.), and here is a short list of things tried to tackle those (which I am calling ``moe variants" for simplicity, the first two are baselines):
- `no_moe`: The **baseline** GPT-2 model with `123.69M`,
- `only_moe`: Every `stride` (here 2, so every second) FFN layer is replaced by expert layers with 8 experts (i.e., 8 identical FFNs) and `top_k=2`. Here the router is just a linear layer `logits = self.w_g(x)` with no noise added to it,
- `+noisy_top_k`: add a data dependent noise to the router to encourage selection of not-so-good experts,
- `+aux_loss`: add load balancing loss to make sure all experts get a fairshare of the load (i.e., total number of tokens in a batch),
- `+z_loss`: Adds another loss term to prevent router logits from becoming too large,
- `+router_full_prec`: router calculation in full precision (`float32`) while other computations in forward pass in bfloat16 precision,
- `+switch_tfm_init`: Also perform switch transformer style initialization of weights on top of everything above, it just adds a scale term during initialization of the MoE layers

I also ran some training with `top_k=1` (selects only 1 expert, identical to `no_moe` in parameter count, but each token can, in theory, choose a different FFN in expert layers), and `top_k=2` (each 
token selects 2 experts) to see how the performance improves/degrades wrt the baseline.

## Hyperparameters
To train the model, you need to run [train_single_gpu.py](https://github.com/swag2198/build-gpt2/blob/main/train_single_gpu.py). The hyperparameters are mostly same as in nanoGPT training and can 
be found here:
```python
@dataclass
class Args:    
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
```


## Compute resources
All training runs are done on an H100-80GB node on the [ferranti](https://portal.mlcloud.uni-tuebingen.de/user-guide/ferranti/ferranti_system_architecture/) cluster at the University of Tübingen 
during my HiWi with Dr. Wieland Brendel at the ELLIS Institute, Tübingen.

## Results

While training, I tracked the validation loss and the accuracy of the HellaSwag evaluation for the models. The goal is to see if the training of MoEs becomes unstable at some point, and if we can
circumvent it by using the training tricks from the literature.

The training was done for 3 random seeds to account for random fluctuations, and the panel below shows
the evolution of the metrics for each seed.

### Q1. How stable/unstable is the MoE training, and do the tricks (aux_loss, z-loss and others described above etc.) help?


**Validation loss plots**:
<p align="center">
  <img src="https://github.com/swag2198/build-gpt2/blob/main/results/figures/val_all_seeds.png?raw=true" alt="validation losses for 3 seeds"/>
</p>


**HellaSwag accuracies**:
<p align="center">
  <img src="https://github.com/swag2198/build-gpt2/blob/main/results/figures/hella_all_seeds.png?raw=true" alt="HellaSwag accuracies for 3 seeds"/>
</p>

From the above, we can clearly make some observations:
- the baseline MoE model with simple router diverges for 2 of the 3 training runs,
- adding noise to the router (yellow curve) helps in stabilizing the training and we don't see divergence for any of the 3 seeds here.
- aux_loss (load balancing loss on top of noisy router, green curve) also shows _somewhat_ stable training, albeit having a little high val loss,
- the other regularizations all seem to result in diverging losses, at least in this model size regime.


### Q2. Do MoEs always outperform the baseline (non-MoE) model?

Here, the goal is to see if we always get performance benefits by replacing the FFN layers with 
multiple experts. Since the MoE models have more (active) parameters than non-MoE models, we would 
expect MoEs to be better. 
The below panel confirms that, and also interestingly shows that if we only choose 1 expert out of 8 (i.e., same active parameters as the non-MoE baseline), the performance is actually worse than the baseline.
We see the stabilizing effect of noisy router here as well.

<p align="center">
  <img src="https://github.com/swag2198/build-gpt2/blob/main/results/figures/moe_vs_no_moe.png?raw=true" alt="val and hella for moe and non-moes"/>
</p>


## References and acknowledgements
The training codes are largely taken from [nanoGPT](https://github.com/karpathy/build-nanogpt), MoE modules are taken from [nanoMoE](https://github.com/wolfecameron/nanoMoE). Thanks a ton to Andrej Karpathy and Cameron R. Wolfe, for providing the starter codes for my experiments!
