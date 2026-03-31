import torch
import numpy as np
import torch.nn as nn
import time
from fused_packed_forward import *

def count_params_and_bytes(model: torch.nn.Module):
    """
    Returns:
      n_params: int
      n_buffers: int
      param_bytes: int
      buffer_bytes: int
      total_bytes: int
    """
    n_params = 0
    param_bytes = 0
    for p in model.parameters():
        n_params += p.numel()
        param_bytes += p.numel() * p.element_size()

    n_buffers = 0
    buffer_bytes = 0
    for b in model.buffers():
        n_buffers += b.numel()
        buffer_bytes += b.numel() * b.element_size()

    total_bytes = param_bytes + buffer_bytes
    return n_params, n_buffers, param_bytes, buffer_bytes, total_bytes


def count_list_params_and_bytes(models):
    """
    Sum params/buffers/bytes over a list of models.
    """
    n_params = n_buffers = 0
    param_bytes = buffer_bytes = total_bytes = 0
    for m in models:
        p, b, pb, bb, tb = count_params_and_bytes(m)
        n_params += p
        n_buffers += b
        param_bytes += pb
        buffer_bytes += bb
        total_bytes += tb
    return n_params, n_buffers, param_bytes, buffer_bytes, total_bytes


def bytes_to_mb(x):  # nicer printing
    return float(x) / (1024.0**2)



class MACCounter:
    def __init__(self):
        self.macs = 0
        self.handles = []

    def add_hooks(self, model: nn.Module):
        def conv_hook(mod: nn.Conv2d, inp, out):
            # out: [B, Cout, Hout, Wout]
            # MACs per output element: (Cin/groups) * Kh * Kw
            x = inp[0]
            B = x.shape[0]
            Cout = out.shape[1]
            Hout = out.shape[2]
            Wout = out.shape[3]
            Cin = mod.in_channels
            Kh, Kw = mod.kernel_size
            groups = mod.groups
            macs = B * Cout * Hout * Wout * (Cin // groups) * Kh * Kw
            self.macs += int(macs)

        def linear_hook(mod: nn.Linear, inp, out):
            # out: [B, Out]
            x = inp[0]
            B = x.shape[0]
            macs = B * mod.out_features * mod.in_features
            self.macs += int(macs)

        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                self.handles.append(m.register_forward_hook(conv_hook))
            elif isinstance(m, nn.Linear):
                self.handles.append(m.register_forward_hook(linear_hook))

    def clear(self):
        self.macs = 0

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


@torch.no_grad()
def estimate_macs(model: nn.Module, x: torch.Tensor, model_type="full"):
    """
    Runs one forward pass to count MACs for Conv2d/Linear.
    Assumes x is the real input shape we benchmark on.
    """
    counter = MACCounter()
    counter.add_hooks(model)
    if model_type == "full":
        _ = model(x, task_id=0)
    else:
        _ = model(x)
    macs = counter.macs
    counter.remove()
    return macs


@torch.no_grad()
def time_callable(fn, iters=100, warmup=20, use_cuda_events=True):
    """
    Times a callable fn() that does the forward pass / prediction.
    Returns mean_ms, std_ms over iters.
    """
    # warmup
    for _ in range(warmup):
        _ = fn()

    if torch.cuda.is_available() and use_cuda_events:
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))  # ms
        return float(np.mean(times)), float(np.std(times))
    else:
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = fn()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
        return float(np.mean(times)), float(np.std(times))


@torch.no_grad()
def peak_cuda_memory_bytes(fn, warmup=5, iters=10):
    """
    Measures peak allocated CUDA memory (bytes) during fn().
    """
    if not torch.cuda.is_available():
        return None
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        _ = fn()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(iters):
        _ = fn()
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated())


def report_protocols(full_model, packed_models, runner, xb, T):
    xb = xb.to(next(full_model.parameters()).device)

    # --- Params / bytes ---
    full_p, full_b, full_pb, full_bb, full_tb = count_params_and_bytes(full_model)
    pack_p, pack_b, pack_pb, pack_bb, pack_tb = count_list_params_and_bytes(packed_models)

    print("\n=== PARAMS / BYTES ===")
    print(f"Full model params: {full_p:,} | bytes: {bytes_to_mb(full_tb):.2f} MB (params+buffers)")
    print(f"Packed models total params: {pack_p:,} | bytes: {bytes_to_mb(pack_tb):.2f} MB (sum over T)")

    # --- MACs (use B=1 and  real spatial size for paper-style per-sample MACs) ---
    x1 = xb[:1].contiguous()
    full_macs = estimate_macs(full_model, x1, model_type="full")

    pack_macs = 0
    for m in packed_models:
        pack_macs += estimate_macs(m, x1, model_type="packed")

    print("\n=== MACs (per-sample, summed) ===")
    print(f"Full model MACs: {full_macs:,}")
    print(f"Packed models MACs (sum over T): {pack_macs:,}")

    # --- Time ---
    print("\n=== TIME (batch) ===")

    # 1) Full sequential 
    mean_ms, std_ms = time_callable(lambda: full_model.class_il_predict_sequential(xb, n_tasks=T), iters=100, warmup=20)
    peak = peak_cuda_memory_bytes(lambda: full_model.class_il_predict_sequential(xb, n_tasks=T))
    print(f"Full sequential (T-pass)  : {mean_ms:.3f} ± {std_ms:.3f} ms | peak mem: {bytes_to_mb(peak):.2f} MB" if peak is not None
          else f"Full sequential (T-pass)  : {mean_ms:.3f} ± {std_ms:.3f} ms")

    # 2) Packed sequential ( existing packed loop)
    mean_ms, std_ms = time_callable(lambda: runner.class_il_predict_sequential(xb), iters=100, warmup=20)
    peak = peak_cuda_memory_bytes(lambda: runner.class_il_predict_sequential(xb))
    print(f"Packed sequential (T-pass): {mean_ms:.3f} ± {std_ms:.3f} ms | peak mem: {bytes_to_mb(peak):.2f} MB" if peak is not None
          else f"Packed sequential (T-pass): {mean_ms:.3f} ± {std_ms:.3f} ms")
    #
    # 3) Packed parallel single call (vmap)
    mean_ms, std_ms = time_callable(lambda: runner.class_il_predict(xb)[0], iters=100, warmup=20)
    peak = peak_cuda_memory_bytes(lambda: runner.class_il_predict(xb)[0])
    print(f"Packed parallel (vmap)    : {mean_ms:.3f} ± {std_ms:.3f} ms | peak mem: {bytes_to_mb(peak):.2f} MB" if peak is not None
          else f"Packed parallel (vmap)    : {mean_ms:.3f} ± {std_ms:.3f} ms")

    # Also helpful: a) forward_all only (pure model compute, no scoring)
    mean_ms, std_ms = time_callable(lambda: runner.forward_all(xb), iters=100, warmup=20)
    print(f"Packed forward_all only   : {mean_ms:.3f} ± {std_ms:.3f} ms")
    #
    # b) Full model task-il
    mean_ms, std_ms = time_callable(lambda: full_model(xb, task_id=0), iters=100, warmup=20)
    peak = peak_cuda_memory_bytes(lambda: full_model(xb, task_id=0))
    print(f"Full task-il (single-pass) : {mean_ms:.3f} ± {std_ms:.3f} ms | peak mem: {bytes_to_mb(peak):.2f} MB" if peak is not None
          else f"Full task-il (single-pass) : {mean_ms:.3f} ± {std_ms:.3f} ms")

    print("\n=== Notes ===")
    print("- MACs reported here are Conv2d/Linear.")
    print("- Peak mem is runtime allocator peak; 'bytes' above is model storage only.")
    print("\n")
    #

