# CUDA Graph Replay Plan

**Current implementation**
- `train_gpt.py:661-749` runs warmup, then iterates the full training loop with eager kernel launches per step.

**Performance impact**
- Re-launching thousands of kernels/collectives from Python each iteration wastes 3-5% step time and adds host-side jitter on 8×H100.

**Optimized solution**
- Capture forward/backward + optimizer step inside a `torch.cuda.CUDAGraph` after warmup, copy fresh inputs/targets/window scalars into static buffers each iteration, then call `graph.replay()`.

**Expected improvement**
- 5-8% faster average step time and tighter latency variance.

**Implementation priority**
- High.

## Relevant code (captured before edits)
```python
model: nn.Module = torch.compile(model, dynamic=False)

########################################
#            Warmup kernels            #
########################################

# Warmup the training kernels, then re-initialize the state so we aren't cheating
warmup_steps = 10
initial_state = dict(model=copy.deepcopy(model.state_dict()),
                     optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers]) # save the initial state
train_loader = distributed_data_generator(args.train_files, world_size * args.train_seq_len, align_to_bos=True)

torch.cuda.reset_peak_memory_stats()
for _ in range(warmup_steps):
    inputs, targets = next(train_loader)
    model(inputs, targets, get_window_size_blocks(1)).backward()
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
model.load_state_dict(initial_state["model"])
for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
    opt.load_state_dict(opt_state)
del train_loader, initial_state
print0(f"memory summary: {torch.cuda.memory_summary(abbreviated=True)}")

########################################
#        Training and validation       #
########################################

train_loader = distributed_data_generator(args.train_files, world_size * args.train_seq_len, align_to_bos=True)
training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.perf_counter()
# begin training
train_steps = args.num_iterations
for step in range(train_steps + 1):
    last_step = (step == train_steps)

    # --------------- VALIDATION SECTION -----------------
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        model.eval()
        val_batch_size = world_size * args.val_seq_len
        assert args.val_tokens % val_batch_size == 0
        val_steps = args.val_tokens // val_batch_size
        val_loader = distributed_data_generator(args.val_files, val_batch_size, align_to_bos=False)
        val_loss = 0
        with torch.no_grad():
            for _ in range(val_steps):
                inputs, targets = next(val_loader)
                val_loss += model(inputs, targets, get_window_size_blocks(step))
        val_loss /= val_steps
        del val_loader
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)
        model.train()
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    if last_step:
        if master_process and args.save_checkpoint:
            log = dict(step=step, code=code, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
            os.makedirs(f"logs/{run_id}", exist_ok=True)
            torch.save(log, f"logs/{run_id}/state_step{step:06d}.pt")
        # the last step only has the validation loop, so break to avoid training
        break

    # --------------- TRAINING SECTION -----------------
    inputs, targets = next(train_loader)
    model(inputs, targets, get_window_size_blocks(step)).backward()
    # set optimization hyperparameters
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * get_lr(step)
    for group in optimizer2.param_groups:
        frac = min(step / 300, 1) # momentum warmup for muon
        group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
    # step the optimizers
    for opt in optimizers:
        opt.step()
    # null the gradients
    model.zero_grad(set_to_none=True)
    # logging
    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms", console=True)
```

---

Goal: remove Python launch overhead by replaying a pre-captured training step on GPU.

How it works: allocate static GPU buffers once, run a single “true” step inside a CUDA Graph, then only copy new data and replay. Shapes, dtypes, and control flow must stay fixed during capture.

High-leverage plan:
	1.	Preallocate static buffers and device scalars

	•	Inputs/targets: same shape/dtype every step.
	•	Scalars: window size, loss scale, and optional lr/momentum multipliers as 0-D CUDA tensors.

	2.	Warm up as you do now to populate caches.
	3.	Capture forward + backward (stage 1). Keep optimizer eager first. This gives most of the win with fewer constraints.
	4.	Optional: move optimizer into the graph (stage 2) only if lr/momentum are constant across replays or are sourced from device tensors via a custom/fused optimizer. Otherwise you must recapture when hyperparameters change.

Drop-in patch (concise):

```python
# ---- after warmup restore ----
train_loader = distributed_data_generator(args.train_files, world_size * args.train_seq_len, align_to_bos=True)

# Static IO buffers (must match per-step shapes/dtypes)
sample_inputs, sample_targets = next(train_loader)
static_inputs  = torch.empty_like(sample_inputs, device="cuda", memory_format=torch.contiguous_format)
static_targets = torch.empty_like(sample_targets, device="cuda", memory_format=torch.contiguous_format)
win_blocks_t   = torch.zeros((), dtype=torch.int64, device="cuda")   # device scalar
loss_t         = torch.zeros((), dtype=torch.float32, device="cuda") # optional for logging

# Optional hyperparam multipliers if you want to vary lr/momentum without recapture
lr_mult_t  = torch.ones((), dtype=torch.float32, device="cuda")
mom_mult_t = torch.ones((), dtype=torch.float32, device="cuda")

g = torch.cuda.CUDAGraph()
static_stream = torch.cuda.Stream()
torch.cuda.synchronize()

# Graph capture: forward + backward (stage 1)
model.train()
model.zero_grad(set_to_none=True)
with torch.cuda.stream(static_stream):
    static_inputs.copy_(sample_inputs, non_blocking=True)
    static_targets.copy_(sample_targets, non_blocking=True)
static_stream.synchronize()

# NOTE: capture requires fixed shapes/allocs inside the region
torch.cuda.synchronize()
with torch.cuda.graph(g):
    out = model(static_inputs, static_targets, win_blocks_t.item())  # if your model reads an int
    # If your model expects a tensor for window blocks, pass win_blocks_t instead of .item()
    loss = out
    loss.backward()
    # Optional in-graph grad scaling to emulate lr schedule:
    # for p in model.parameters(): 
    #     if p.grad is not None: p.grad.mul_(lr_mult_t)

# ----------------- training loop -----------------
training_time_ms = 0
torch.cuda.synchronize(); t0 = time.perf_counter()
for step in range(args.num_iterations + 1):
    last_step = (step == args.num_iterations)

    # ------- validation (unchanged) -------
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        model.eval()
        # ... existing validation block ...
        model.train()
        torch.cuda.synchronize(); t0 = time.perf_counter()

    if last_step:
        if master_process and args.save_checkpoint:
            log = dict(step=step, code=code, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
            os.makedirs(f"logs/{run_id}", exist_ok=True)
            torch.save(log, f"logs/{run_id}/state_step{step:06d}.pt")
        break

    # ------- TRAIN -------
    # 1) copy fresh data into static buffers
    inputs, targets = next(train_loader)
    static_inputs.copy_(inputs, non_blocking=True)
    static_targets.copy_(targets, non_blocking=True)
    # update device scalars
    win_blocks_t.fill_(get_window_size_blocks(step))
    # if emulating lr/momentum schedules inside graph:
    # lr_mult_t.fill_(get_lr(step) / base_lr)
    # mom_mult_t.fill_( ... )

    # 2) replay captured fwd+bwd
    g.replay()

    # 3) host-side hyperparams and optimizer step (stage 1)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * get_lr(step)
    for group in optimizer2.param_groups:
        frac = min(step / 300, 1)
        group["momentum"] = (1 - frac) * 0.85 + frac * 0.95

    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)

    # logging (unchanged)
    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(f"step:{step+1}/{args.num_iterations} train_time:{approx_training_time_ms:.0f}ms "
           f"step_avg:{approx_training_time_ms/(step+1):.2f}ms", console=True)

If you insist on optimizer inside the graph (stage 2):
	•	Constraint: any Python-read hyperparameter becomes constant at capture. To vary lr/momentum you need either:
	1.	Recapture at schedule change points (few times per run), or
	2.	A graph-friendly optimizer that reads lr/momentum from device tensors you update each step, or
	3.	Emulate lr schedule by scaling gradients inside the graph via a device scalar lr_mult_t and keep a fixed base lr in the captured optimizer.
	•	Capture pattern:

# before capture: set base lr once in param_groups; momentum fixed
with torch.cuda.graph(g):
    out = model(static_inputs, static_targets, win_blocks_t)
    loss = out
    loss.backward()
    # grad scaling to vary effective lr
    # for p in model.parameters():
    #     if p.grad is not None: p.grad.mul_(lr_mult_t)
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)

Key constraints and pitfalls:
	•	Shapes and dtypes must not change across replays. Keep batch size, seq len, and model eval path constant.
	•	No new memory allocations inside capture. Pre-create work buffers. set_to_none=True avoids allocator churn.
	•	RNG: if you use dropout or CUDA RNG, capture fixes the RNG state progression. That is fine for training if you accept deterministic replay per iteration. To change it, bump a device counter or recapture.
	•	DDP: world size and communication topology must be constant. Avoid dynamic barriers. NCCL calls are fine when shapes are static.
	•	AMP: keep GradScaler out of the graph or make its scale a device scalar you manage manually. Simplest is static loss scale.
	•	Validation: keep it eager. Do not share graph buffers.

Measurement:
	•	Report mean and p95 step time over N>200 steps with and without graphs. Disable noisy logging. Expect 5–8% average improvement on 8×H100 and tighter variance. Use CUDA events for device-side timing if you want host jitter removed.

Rollout order:
	1.	Stage 1 now: graph forward+backward. Keep optimizer eager. Low risk. Gains ~3–6%.
	2.	Stage 2 later: move optimizer into graph with gradient scaling or periodic recapture. Push to 5–8%+.

Priority: high.