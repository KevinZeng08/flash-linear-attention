#!/usr/bin/env python3
"""
Profile script for KDA (Key-Dependent Attention) with Context Parallelism (CP).
Measures execution latency of chunk_kda with CP, supporting forward-only and forward+backward profiling.

Combines the profiling style of profile_kda.py with the CP distributed setup from
benchmarks/cp/benchmark_kda_cp8_vs_cp2tp.py.

Usage:
    # CP8 forward only, 128k sequence
    torchrun --nproc_per_node=8 profile_kda_cp.py --mode fwd --T 131072 --cp-size 8

    # CP8 forward + backward, 128k sequence
    torchrun --nproc_per_node=8 profile_kda_cp.py --mode fwdbwd --T 131072 --cp-size 8

    # CP8 forward + backward with torch profiler and chrome trace export
    torchrun --nproc_per_node=8 profile_kda_cp.py --mode fwdbwd --T 131072 --cp-size 8 --profiler

    # CP4 with TP2, 128k sequence
    torchrun --nproc_per_node=8 profile_kda_cp.py --mode fwdbwd --T 131072 --cp-size 4 --tp-size 2

    # CP8 with single-GPU baseline comparison
    torchrun --nproc_per_node=8 profile_kda_cp.py --mode fwdbwd --T 131072 --cp-size 8 --with-baseline

    # CP8 with kernel-level profiling (rank 0 only)
    torchrun --nproc_per_node=8 profile_kda_cp.py --mode fwdbwd --T 131072 --cp-size 8 --profile-kernels

    # CP8 varlen mode (variable-length sequences packed into a single batch)
    torchrun --nproc_per_node=8 profile_kda_cp.py --mode fwdbwd --T 131072 --cp-size 8 --varlen --num-seqs 18

    # Sweep over multiple sequence lengths
    torchrun --nproc_per_node=8 profile_kda_cp.py --sweep --cp-size 8
"""

import argparse
import os
import random
import time

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile, record_function, schedule, tensorboard_trace_handler

from fla.ops.cp import build_cp_context
from fla.ops.kda import chunk_kda

# Configuration
DTYPE = torch.bfloat16


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def exclusive_cumsum(a: list[int]):
    r = [0]
    for v in a:
        r.append(r[-1] + v)
    return r


def generate_random_seq_lens(
    num_seqs: int,
    total_len: int,
    min_seq_len: int = 63,
    variance: float = 1.0,
    seed: int = 42,
) -> list:
    """Generate random sequence lengths that sum to total_len with each >= min_seq_len."""
    assert total_len >= num_seqs * min_seq_len, \
        f"total_len ({total_len}) must be >= num_seqs ({num_seqs}) * min_seq_len ({min_seq_len})"

    set_seed(seed)

    base_len = total_len // num_seqs
    remainder = total_len % num_seqs

    if variance == 0.0:
        seq_lens = [base_len] * num_seqs
        for i in range(remainder):
            seq_lens[i] += 1
    else:
        # Give each sequence the minimum length, then distribute the rest randomly
        seq_lens = [min_seq_len] * num_seqs
        remaining = total_len - num_seqs * min_seq_len
        if remaining > 0:
            # Dirichlet-like random split
            weights = [random.random() ** variance for _ in range(num_seqs)]
            total_w = sum(weights)
            for i in range(num_seqs):
                extra = int(remaining * weights[i] / total_w)
                seq_lens[i] += extra
            # Distribute rounding error
            diff = total_len - sum(seq_lens)
            for i in range(abs(diff)):
                seq_lens[i % num_seqs] += 1 if diff > 0 else -1

    # Enforce min_seq_len
    for i in range(num_seqs):
        if seq_lens[i] < min_seq_len:
            deficit = min_seq_len - seq_lens[i]
            seq_lens[i] = min_seq_len
            for j in range(num_seqs):
                if j != i and seq_lens[j] > min_seq_len + deficit:
                    seq_lens[j] -= deficit
                    break

    assert sum(seq_lens) == total_len, f"sum(seq_lens)={sum(seq_lens)} != total_len={total_len}"
    assert all(s >= min_seq_len for s in seq_lens), "Some seq_len < min_seq_len"
    return seq_lens


def print_rank0(*args, **kwargs):
    """Print only on rank 0, with barrier for synchronization."""
    if dist.is_initialized():
        dist.barrier()
        if dist.get_rank() == 0:
            print(*args, **kwargs)
        dist.barrier()
    else:
        print(*args, **kwargs)


def all_gather(x, group=None) -> torch.Tensor:
    world_size = dist.get_world_size(group=group)
    y = torch.empty(world_size * x.size(0), *x.shape[1:], device=x.device, dtype=x.dtype)
    dist.all_gather_into_tensor(y, x, group=group)
    return y


def warmup_gpu(device, num_iters=10):
    """Warmup GPU to ensure stable measurements."""
    for _ in range(num_iters):
        a = torch.randn(1024, 1024, device=device)
        b = torch.randn(1024, 1024, device=device)
        torch.matmul(a, b)
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Input creation helpers
# ---------------------------------------------------------------------------

def create_input_data(B, T, H, K, V, device, dtype=DTYPE, requires_grad=True):
    """Create input tensors for KDA."""
    torch.manual_seed(42 + dist.get_rank() if dist.is_initialized() else 42)

    q = torch.randn(B, T, H, K, dtype=dtype, device=device)
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    v = torch.randn(B, T, H, V, dtype=dtype, device=device)
    g = torch.randn(B, T, H, K, dtype=dtype, device=device)
    beta = torch.rand(B, T, H, dtype=dtype, device=device).sigmoid()

    A_log = torch.randn(H, device=device, dtype=torch.float32)
    dt_bias = torch.randn(H * V, device=device, dtype=torch.float32)

    if requires_grad:
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        g.requires_grad_(True)
        beta.requires_grad_(True)

    do = torch.randn(B, T, H, V, dtype=dtype, device=device) if requires_grad else None
    return q, k, v, g, beta, A_log, dt_bias, do


# ---------------------------------------------------------------------------
# Kernel profiling helpers (from benchmark_kda_cp8_vs_cp2tp.py)
# ---------------------------------------------------------------------------

def profile_kernels_fn(fn, steps=5, warmup=2):
    """Profile individual CUDA kernels using PyTorch profiler."""
    for _ in range(warmup):
        fn()

    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=0, warmup=1, active=steps),
        with_stack=False,
        record_shapes=False,
        profile_memory=False,
    ) as prof:
        for _ in range(steps + 1):
            torch.cuda.synchronize()
            fn()
            torch.cuda.synchronize()
            prof.step()

    kernel_stats = []
    for event in prof.key_averages():
        if event.cuda_time_total > 0:
            kernel_stats.append({
                'name': event.key,
                'cuda_time_ms': event.cuda_time_total / 1000,
                'calls': event.count,
            })
    kernel_stats.sort(key=lambda x: x['cuda_time_ms'], reverse=True)
    return kernel_stats


def format_kernel_table(kernel_stats, top_n=20):
    """Format kernel stats as a table."""
    if not kernel_stats:
        return "No kernel data available"

    total_time = sum(k['cuda_time_ms'] for k in kernel_stats)
    lines = []
    lines.append(f"{'Rank':<4} {'Kernel Name':<60} {'Time (ms)':<12} {'Calls':<8} {'%':<6}")
    lines.append("-" * 100)
    for i, stat in enumerate(kernel_stats[:top_n]):
        pct = stat['cuda_time_ms'] / total_time * 100 if total_time > 0 else 0
        name = stat['name'][:58] if len(stat['name']) > 58 else stat['name']
        lines.append(f"{i+1:<4} {name:<60} {stat['cuda_time_ms']:<12.3f} {stat['calls']:<8} {pct:<6.1f}")
    lines.append("-" * 100)
    lines.append(f"{'Total':<65} {total_time:<12.3f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Distributed setup helpers
# ---------------------------------------------------------------------------

def setup_distributed_groups(cp_size, tp_size):
    """Create CP and TP process groups. Returns (cp_group, tp_group, cp_rank, tp_rank)."""
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    assert cp_size * tp_size == world_size, \
        f"CP size ({cp_size}) * TP size ({tp_size}) must equal world size ({world_size})"

    cp_rank = rank // tp_size
    tp_rank = rank % tp_size

    # CP group: ranks that share the same TP rank
    cp_ranks = list(range(tp_rank, world_size, tp_size))
    # TP group: ranks that share the same CP rank
    tp_ranks = list(range(cp_rank * tp_size, (cp_rank + 1) * tp_size))

    cp_group = dist.new_group(cp_ranks)
    tp_group = dist.new_group(tp_ranks)

    return cp_group, tp_group, cp_rank, tp_rank


# ---------------------------------------------------------------------------
# Profile: forward only with CP
# ---------------------------------------------------------------------------

def profile_kda_cp_fwd(
    B, T, H, K, V,
    cp_size, tp_size,
    cp_group, tp_group, cp_rank, tp_rank,
    varlen=False,
    variance=1.0,
    num_seqs=18,
    num_warmup=10,
    num_iters=20,
    with_profiler=False,
    profile_kernels_flag=False,
    with_baseline=False,
):
    """Profile KDA forward pass with context parallelism."""
    device = torch.cuda.current_device()
    rank = dist.get_rank()

    T_local = T // cp_size
    H_local = H // tp_size

    print_rank0(f"\n{'='*60}")
    print_rank0(f"Profiling KDA CP Forward: B={B}, T_total={T}, T_local={T_local}")
    print_rank0(f"H={H}, H_local={H_local}, K={K}, V={V}")
    print_rank0(f"CP_size={cp_size}, TP_size={tp_size}")
    print_rank0(f"varlen={varlen}, variance={variance}")
    print_rank0(f"{'='*60}")

    # Build cu_seqlens (global, before CP partitioning)
    if varlen:
        seqlens = generate_random_seq_lens(num_seqs=num_seqs, total_len=T, min_seq_len=63, variance=variance, seed=42)
        cum_seqlens = exclusive_cumsum(seqlens)
        cu_seqlens_global = torch.tensor(cum_seqlens, dtype=torch.int32, device=device)
        print_rank0(f"Varlen seqlens ({num_seqs} seqs): {seqlens}")
    else:
        cu_seqlens_global = torch.tensor([0, T], dtype=torch.int32, device=device)

    # Build CP context
    cp_context = build_cp_context(cu_seqlens_global, cp_group)
    cp_cu_seqlens = cp_context.cu_seqlens

    if rank == 0:
        print(f"[Rank {rank}] cp_context: cu_seqlens={cp_context.cu_seqlens.tolist()}, "
              f"is_first_rank={cp_context.is_first_rank}, is_last_rank={cp_context.is_last_rank}, "
              f"pre_num_ranks={cp_context.pre_num_ranks}, post_num_ranks={cp_context.post_num_ranks}")

    # Create local input tensors
    q, k, v, g, beta, A_log, dt_bias, _ = create_input_data(
        B, T_local, H_local, K, V, device, requires_grad=False
    )

    # Define CP forward function
    def kda_cp_fwd():
        with torch.no_grad():
            o, _ = chunk_kda(
                q=q, k=k, v=v, g=g, beta=beta,
                A_log=A_log, dt_bias=dt_bias,
                use_gate_in_kernel=True,
                safe_gate=True,
                lower_bound=-5,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cp_cu_seqlens,
                cp_context=cp_context,
            )
            if tp_size > 1:
                dist.all_reduce(o, group=tp_group)
            dist.barrier()
        return o

    # Warmup
    print_rank0(f"Warming up ({num_warmup} iterations)...")
    warmup_gpu(device, num_warmup)
    for _ in range(num_warmup):
        kda_cp_fwd()
    torch.cuda.synchronize()

    # Kernel-level profiling (rank 0 only)
    if profile_kernels_flag and rank == 0:
        print(f"\n--- Kernel Profiling (Rank 0) ---")
        kernel_stats = profile_kernels_fn(kda_cp_fwd, steps=5, warmup=2)
        print(format_kernel_table(kernel_stats, top_n=20))
        print()

    # Torch profiler
    if with_profiler:
        print_rank0(f"\nRunning torch.profiler ({num_iters} iterations)...")
        trace_dir = f"./log/kda_cp{cp_size}_tp{tp_size}_fwd_profile"
        os.makedirs(trace_dir, exist_ok=True) if rank == 0 else None

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=1, warmup=2, active=num_iters),
            record_shapes=False,
            with_stack=False,
        ) as prof:
            for _ in range(num_iters + 3):
                with record_function("chunk_kda_cp_fwd"):
                    kda_cp_fwd()
                prof.step()

        torch.cuda.synchronize()

        # Export chrome trace on rank 1
        # Because in first rank or last rank, 
        # the preprocess kernel of CP is only called once
        if rank == 1:
            trace_path = f"rank_{rank}_kda_cp{cp_size}_tp{tp_size}_fwd{'_varlen_var' + str(variance) if varlen else ''}_trace.json"
            prof.export_chrome_trace(trace_path)
            print(f"Chrome trace saved to: {trace_path}")

    # Simple timing measurement
    print_rank0(f"\n--- Simple Timing ({num_iters} iterations) ---")
    times = []
    for _ in range(num_iters):
        torch.cuda.synchronize()
        dist.barrier()
        start = time.perf_counter()
        kda_cp_fwd()
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print_rank0(f"CP Forward: Avg={avg_time:.3f}ms, Min={min_time:.3f}ms, Max={max_time:.3f}ms")

    # Baseline comparison: single-GPU with local sequence length
    if with_baseline:
        T_baseline = T_local
        q_b, k_b, v_b, g_b, beta_b, A_log_b, dt_bias_b, _ = create_input_data(
            B, T_baseline, H_local, K, V, device, requires_grad=False
        )

        def kda_baseline_fwd():
            with torch.no_grad():
                o, _ = chunk_kda(
                    q=q_b, k=k_b, v=v_b, g=g_b, beta=beta_b,
                    A_log=A_log_b, dt_bias=dt_bias_b,
                    use_gate_in_kernel=True,
                    safe_gate=True,
                    lower_bound=-5,
                    use_qk_l2norm_in_kernel=True,
                )
            return o

        # Warmup baseline
        for _ in range(num_warmup):
            kda_baseline_fwd()
        torch.cuda.synchronize()

        base_times = []
        for _ in range(num_iters):
            torch.cuda.synchronize()
            start = time.perf_counter()
            kda_baseline_fwd()
            torch.cuda.synchronize()
            end = time.perf_counter()
            base_times.append((end - start) * 1000)

        avg_base = sum(base_times) / len(base_times)
        scaled_base = avg_base * cp_size

        if rank == 0:
            print(f"\n--- Baseline Comparison ---")
            print(f"Single-GPU local ({T_baseline}): Avg={avg_base:.3f}ms")
            print(f"Scaled to full seq ({T}):     Est={scaled_base:.3f}ms")
            print(f"CP Speedup vs Scaled:          {scaled_base / avg_time:.2f}x")

    # Memory summary
    if rank == 0:
        print(f"\n--- Memory (Rank 0) ---")
        print(f"Allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        print(f"Max allocated: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB")
        print(f"Reserved: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")

    return avg_time


# ---------------------------------------------------------------------------
# Profile: forward + backward with CP
# ---------------------------------------------------------------------------

def profile_kda_cp_fwd_bwd(
    B, T, H, K, V,
    cp_size, tp_size,
    cp_group, tp_group, cp_rank, tp_rank,
    varlen=False,
    variance=1.0,
    num_seqs=18,
    num_warmup=10,
    num_iters=20,
    with_profiler=False,
    profile_kernels_flag=False,
    with_baseline=False,
):
    """Profile KDA forward + backward with context parallelism."""
    device = torch.cuda.current_device()
    rank = dist.get_rank()

    T_local = T // cp_size
    H_local = H // tp_size

    print_rank0(f"\n{'='*60}")
    print_rank0(f"Profiling KDA CP FWD+BWD: B={B}, T_total={T}, T_local={T_local}")
    print_rank0(f"H={H}, H_local={H_local}, K={K}, V={V}")
    print_rank0(f"CP_size={cp_size}, TP_size={tp_size}")
    print_rank0(f"varlen={varlen}, variance={variance}")
    print_rank0(f"{'='*60}")

    # Build cu_seqlens
    if varlen:
        seqlens = generate_random_seq_lens(num_seqs=num_seqs, total_len=T, min_seq_len=63, variance=variance, seed=42)
        cum_seqlens = exclusive_cumsum(seqlens)
        cu_seqlens_global = torch.tensor(cum_seqlens, dtype=torch.int32, device=device)
        print_rank0(f"Varlen seqlens ({num_seqs} seqs): {seqlens}")
    else:
        cu_seqlens_global = torch.tensor([0, T], dtype=torch.int32, device=device)

    # Build CP context
    cp_context = build_cp_context(cu_seqlens_global, cp_group)
    cp_cu_seqlens = cp_context.cu_seqlens

    if rank == 0:
        print(f"[Rank {rank}] cp_context: cu_seqlens={cp_context.cu_seqlens.tolist()}, "
              f"is_first_rank={cp_context.is_first_rank}, is_last_rank={cp_context.is_last_rank}, "
              f"pre_num_ranks={cp_context.pre_num_ranks}, post_num_ranks={cp_context.post_num_ranks}")

    # Create local input tensors (with grad)
    q, k, v, g, beta, A_log, dt_bias, do = create_input_data(
        B, T_local, H_local, K, V, device, requires_grad=True
    )

    # Define CP forward+backward function
    def kda_cp_fwd_bwd():
        q_grad = q.clone().requires_grad_(True)
        k_grad = k.clone().requires_grad_(True)
        v_grad = v.clone().requires_grad_(True)
        g_grad = g.clone().requires_grad_(True)
        beta_grad = beta.clone().requires_grad_(True)

        o, _ = chunk_kda(
            q=q_grad, k=k_grad, v=v_grad, g=g_grad, beta=beta_grad,
            A_log=A_log, dt_bias=dt_bias,
            use_gate_in_kernel=True,
            safe_gate=True,
            lower_bound=-5,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cp_cu_seqlens,
            cp_context=cp_context,
        )
        if tp_size > 1:
            dist.all_reduce(o, group=tp_group)
        dist.barrier()
        o.backward(do)
        dist.barrier()
        return o

    # Warmup
    print_rank0(f"Warming up ({num_warmup} iterations)...")
    warmup_gpu(device, num_warmup)
    for _ in range(num_warmup):
        kda_cp_fwd_bwd()
    torch.cuda.synchronize()

    # Kernel-level profiling (rank 0 only)
    if profile_kernels_flag and rank == 0:
        print(f"\n--- Kernel Profiling (Rank 0) ---")
        kernel_stats = profile_kernels_fn(kda_cp_fwd_bwd, steps=5, warmup=2)
        print(format_kernel_table(kernel_stats, top_n=25))
        print()

    # Torch profiler
    if with_profiler:
        print_rank0(f"\nRunning torch.profiler ({num_iters} iterations)...")
        trace_dir = f"./log/kda_cp{cp_size}_tp{tp_size}_fwdbwd_profile"
        os.makedirs(trace_dir, exist_ok=True) if rank == 0 else None

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=1, warmup=2, active=num_iters),
            record_shapes=False,
            with_stack=False,
        ) as prof:
            for _ in range(num_iters + 3):
                with record_function("chunk_kda_cp_fwd_bwd"):
                    kda_cp_fwd_bwd()
                prof.step()

        torch.cuda.synchronize()

        # Export chrome trace on rank 1
        # Because in first rank or last rank, 
        # the preprocess kernel of CP is only called once
        if rank == 1:
            trace_path = f"rank_{rank}_kda_cp{cp_size}_tp{tp_size}_fwdbwd{'_varlen_var' + str(variance) if varlen else ''}_trace.json"
            prof.export_chrome_trace(trace_path)
            print(f"Chrome trace saved to: {trace_path}")

    # Simple timing measurement with fwd/bwd split
    print_rank0(f"\n--- Simple Timing ({num_iters} iterations) ---")
    times = []
    fwd_times = []
    bwd_times = []

    for _ in range(num_iters):
        q_grad = q.clone().requires_grad_(True)
        k_grad = k.clone().requires_grad_(True)
        v_grad = v.clone().requires_grad_(True)
        g_grad = g.clone().requires_grad_(True)
        beta_grad = beta.clone().requires_grad_(True)

        torch.cuda.synchronize()
        dist.barrier()
        start = time.perf_counter()

        o, _ = chunk_kda(
            q=q_grad, k=k_grad, v=v_grad, g=g_grad, beta=beta_grad,
            A_log=A_log, dt_bias=dt_bias,
            use_gate_in_kernel=True,
            safe_gate=True,
            lower_bound=-5,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cp_cu_seqlens,
            cp_context=cp_context,
        )
        if tp_size > 1:
            dist.all_reduce(o, group=tp_group)
        dist.barrier()
        torch.cuda.synchronize()
        fwd_end = time.perf_counter()

        o.backward(do)
        dist.barrier()
        torch.cuda.synchronize()
        bwd_end = time.perf_counter()

        fwd_times.append((fwd_end - start) * 1000)
        bwd_times.append((bwd_end - fwd_end) * 1000)
        times.append((bwd_end - start) * 1000)

    avg_time = sum(times) / len(times)
    avg_fwd = sum(fwd_times) / len(fwd_times)
    avg_bwd = sum(bwd_times) / len(bwd_times)

    print_rank0(f"CP Forward:  Avg={avg_fwd:.3f}ms")
    print_rank0(f"CP Backward: Avg={avg_bwd:.3f}ms")
    print_rank0(f"CP Total:    Avg={avg_time:.3f}ms, Min={min(times):.3f}ms, Max={max(times):.3f}ms")

    # Baseline comparison
    if with_baseline:
        T_baseline = T_local
        q_b, k_b, v_b, g_b, beta_b, A_log_b, dt_bias_b, do_b = create_input_data(
            B, T_baseline, H_local, K, V, device, requires_grad=True
        )

        def kda_baseline_fwd_bwd():
            qb = q_b.clone().requires_grad_(True)
            kb = k_b.clone().requires_grad_(True)
            vb = v_b.clone().requires_grad_(True)
            gb = g_b.clone().requires_grad_(True)
            bb = beta_b.clone().requires_grad_(True)
            o, _ = chunk_kda(
                q=qb, k=kb, v=vb, g=gb, beta=bb,
                A_log=A_log_b, dt_bias=dt_bias_b,
                use_gate_in_kernel=True,
                safe_gate=True,
                lower_bound=-5,
                use_qk_l2norm_in_kernel=True,
            )
            o.backward(do_b)
            return o

        for _ in range(num_warmup):
            kda_baseline_fwd_bwd()
        torch.cuda.synchronize()

        base_times = []
        base_fwd_times = []
        base_bwd_times = []
        for _ in range(num_iters):
            qb = q_b.clone().requires_grad_(True)
            kb = k_b.clone().requires_grad_(True)
            vb = v_b.clone().requires_grad_(True)
            gb = g_b.clone().requires_grad_(True)
            bb = beta_b.clone().requires_grad_(True)

            torch.cuda.synchronize()
            start = time.perf_counter()
            o, _ = chunk_kda(
                q=qb, k=kb, v=vb, g=gb, beta=bb,
                A_log=A_log_b, dt_bias=dt_bias_b,
                use_gate_in_kernel=True,
                safe_gate=True,
                lower_bound=-5,
                use_qk_l2norm_in_kernel=True,
            )
            torch.cuda.synchronize()
            fwd_end = time.perf_counter()
            o.backward(do_b)
            torch.cuda.synchronize()
            bwd_end = time.perf_counter()

            base_fwd_times.append((fwd_end - start) * 1000)
            base_bwd_times.append((bwd_end - fwd_end) * 1000)
            base_times.append((bwd_end - start) * 1000)

        avg_base = sum(base_times) / len(base_times)
        avg_base_fwd = sum(base_fwd_times) / len(base_fwd_times)
        avg_base_bwd = sum(base_bwd_times) / len(base_bwd_times)
        scaled_base = avg_base * cp_size

        if rank == 0:
            print(f"\n--- Baseline Comparison ---")
            print(f"Single-GPU local ({T_baseline}):")
            print(f"  Forward:  Avg={avg_base_fwd:.3f}ms")
            print(f"  Backward: Avg={avg_base_bwd:.3f}ms")
            print(f"  Total:    Avg={avg_base:.3f}ms")
            print(f"Scaled to full seq ({T}): Est={scaled_base:.3f}ms")
            print(f"CP Speedup vs Scaled:     {scaled_base / avg_time:.2f}x")
            print(f"  Forward speedup:        {(avg_base_fwd * cp_size) / avg_fwd:.2f}x")
            print(f"  Backward speedup:       {(avg_base_bwd * cp_size) / avg_bwd:.2f}x")

    # Memory summary
    if rank == 0:
        print(f"\n--- Memory (Rank 0) ---")
        print(f"Allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        print(f"Max allocated: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB")
        print(f"Reserved: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")

    return avg_time, avg_fwd, avg_bwd


# ---------------------------------------------------------------------------
# Sweep over configurations
# ---------------------------------------------------------------------------

def sweep_configs(args, cp_group, tp_group, cp_rank, tp_rank):
    """Run profiling across multiple configurations."""
    configs = [
        # (B, T, H, K, V)
        (1, 32768,  32, 128, 128),
        (1, 65536,  32, 128, 128),
        (1, 131072, 32, 128, 128),
        (1, 262144, 32, 128, 128),
        (1, 131072, 64, 128, 128),
        (1, 131072, 32, 256, 256),
    ]

    results = []
    for B, T, H, K, V in configs:
        # Validate T is divisible by cp_size
        if T % args.cp_size != 0:
            print_rank0(f"Skipping T={T}: not divisible by cp_size={args.cp_size}")
            continue

        try:
            if args.mode == 'fwd':
                fwd_time = profile_kda_cp_fwd(
                    B, T, H, K, V,
                    args.cp_size, args.tp_size,
                    cp_group, tp_group, cp_rank, tp_rank,
                    varlen=args.varlen,
                    variance=args.variance,
                    num_seqs=args.num_seqs,
                    num_warmup=args.warmup,
                    num_iters=args.iters,
                    with_baseline=args.with_baseline,
                )
                results.append((B, T, H, K, V, fwd_time, 0, 0))
            else:
                avg_time, avg_fwd, avg_bwd = profile_kda_cp_fwd_bwd(
                    B, T, H, K, V,
                    args.cp_size, args.tp_size,
                    cp_group, tp_group, cp_rank, tp_rank,
                    varlen=args.varlen,
                    variance=args.variance,
                    num_seqs=args.num_seqs,
                    num_warmup=args.warmup,
                    num_iters=args.iters,
                    with_baseline=args.with_baseline,
                )
                results.append((B, T, H, K, V, avg_fwd, avg_bwd, avg_time))
        except Exception as e:
            print_rank0(f"Error with config (B={B}, T={T}, H={H}, K={K}, V={V}): {e}")
            import traceback
            if dist.get_rank() == 0:
                traceback.print_exc()

    # Print summary
    if dist.get_rank() == 0:
        print(f"\n{'='*90}")
        print(f"Sweep Summary (CP{args.cp_size} TP{args.tp_size}, mode={args.mode})")
        print(f"{'='*90}")
        print(f"{'B':>4} {'T':>8} {'H':>4} {'K':>4} {'V':>4} {'Fwd(ms)':>10} {'Bwd(ms)':>10} {'Total(ms)':>10}")
        print("-" * 90)
        for B, T, H, K, V, fwd, bwd, total in results:
            print(f"{B:>4} {T:>8} {H:>4} {K:>4} {V:>4} {fwd:>10.3f} {bwd:>10.3f} {total:>10.3f}")
        print(f"{'='*90}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Profile KDA with Context Parallelism (CP)')
    parser.add_argument('--mode', type=str, default='fwdbwd', choices=['fwd', 'fwdbwd'],
                        help='Profiling mode: forward only, or forward + backward')
    parser.add_argument('--B', type=int, default=1, help='Batch size')
    parser.add_argument('--T', type=int, default=131072, help='Total sequence length (default: 128k)')
    parser.add_argument('--H', type=int, default=32, help='Number of heads')
    parser.add_argument('--K', type=int, default=128, help='Key dimension')
    parser.add_argument('--V', type=int, default=128, help='Value dimension')

    # CP / TP configuration
    parser.add_argument('--cp-size', type=int, default=None,
                        help='Context parallel size (default: world_size)')
    parser.add_argument('--tp-size', type=int, default=1, help='Tensor parallel size (default: 1)')

    # Varlen
    parser.add_argument('--varlen', action='store_true',
                        help='Use variable-length sequences packed into a single batch')
    parser.add_argument('--num-seqs', type=int, default=18,
                        help='Number of sequences for varlen mode (default: 18)')
    parser.add_argument('--variance', type=float, default=1.0,
                        help='Variance for random sequence length generation')

    # Profiling options
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup iterations')
    parser.add_argument('--iters', type=int, default=20, help='Number of profiling iterations')
    parser.add_argument('--profiler', action='store_true', help='Use torch.profiler (with trace export)')
    parser.add_argument('--profile-kernels', action='store_true',
                        help='Profile individual CUDA kernels (rank 0 only)')
    parser.add_argument('--with-baseline', action='store_true',
                        help='Compare with single-GPU baseline (local seqlen, no CP)')

    # Sweep
    parser.add_argument('--sweep', action='store_true',
                        help='Run sweep over multiple configurations')

    args = parser.parse_args()

    # Initialize distributed
    dist.init_process_group()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    torch.manual_seed(rank + 42)
    torch.cuda.manual_seed(rank + 42)
    random.seed(42)
    torch.cuda.set_device(local_rank)

    # Default cp_size = world_size
    if args.cp_size is None:
        args.cp_size = world_size // args.tp_size

    # Validate
    assert args.cp_size * args.tp_size == world_size, \
        f"cp_size ({args.cp_size}) * tp_size ({args.tp_size}) must equal world_size ({world_size})"
    assert args.T % args.cp_size == 0, \
        f"T ({args.T}) must be divisible by cp_size ({args.cp_size})"
    assert args.H % args.tp_size == 0, \
        f"H ({args.H}) must be divisible by tp_size ({args.tp_size})"

    # Print environment info
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"KDA CP Profiling")
        print(f"{'='*60}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA device: {torch.cuda.get_device_name(local_rank)}")
        print(f"World size: {world_size}")
        print(f"CP size: {args.cp_size}, TP size: {args.tp_size}")
        print(f"{'='*60}\n")

    # Create log directory
    os.makedirs('./log', exist_ok=True) if rank == 0 else None

    # Setup distributed groups
    cp_group, tp_group, cp_rank, tp_rank = setup_distributed_groups(args.cp_size, args.tp_size)

    if args.sweep:
        sweep_configs(args, cp_group, tp_group, cp_rank, tp_rank)
    else:
        if args.mode == 'fwd':
            profile_kda_cp_fwd(
                args.B, args.T, args.H, args.K, args.V,
                args.cp_size, args.tp_size,
                cp_group, tp_group, cp_rank, tp_rank,
                varlen=args.varlen,
                variance=args.variance,
                num_seqs=args.num_seqs,
                num_warmup=args.warmup,
                num_iters=args.iters,
                with_profiler=args.profiler,
                profile_kernels_flag=args.profile_kernels,
                with_baseline=args.with_baseline,
            )
        else:
            profile_kda_cp_fwd_bwd(
                args.B, args.T, args.H, args.K, args.V,
                args.cp_size, args.tp_size,
                cp_group, tp_group, cp_rank, tp_rank,
                varlen=args.varlen,
                variance=args.variance,
                num_seqs=args.num_seqs,
                num_warmup=args.warmup,
                num_iters=args.iters,
                with_profiler=args.profiler,
                profile_kernels_flag=args.profile_kernels,
                with_baseline=args.with_baseline,
            )

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
