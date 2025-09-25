# Window-Scalar Caching Plan

**Current implementation**
- `train_gpt.py:640-659` re-creates a pinned tensor and issues a device transfer for every call to `get_window_size_blocks`.

**Performance impact**
- Adds 1-3% step overhead and risks Dynamo recompiles by materializing new CUDA tensors each iteration.

**Optimized solution**
- Precompute the small set of sliding-window values once (e.g., table on GPU) and index by step progress, or convert schedule to plain integers consumed inside the CUDA graph.

**Expected improvement**
- 1-3% faster iterations plus fewer graph breaks.

**Implementation priority**
- Medium.

## Relevant code (captured before edits)
```python
def get_lr(step: int):
    x = step / args.num_iterations # progress in training
    assert 0 <= x < 1
    if x < 1 - args.cooldown_frac:
        return 1.0
    else:
        w = (1 - x) / args.cooldown_frac
        return w * 1.0 + (1 - w) * 0.1

# attention window size schedule: linearly increase
@lru_cache(1)
def get_window_size_blocks_helper(window_size: int):
    return torch.tensor(window_size // 128, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
def get_window_size_blocks(step: int):
    x = step / args.num_iterations # progress in training
    assert 0 <= x <= 1
    # Linearly increase the block-wise sliding window size over training 128 -> 1792
    # increase by @fernbear.bsky.social; block-wise by @YouJiacheng
    window_size = next_multiple_of_n(1728 * x, n=128)
    return get_window_size_blocks_helper(window_size)
```
