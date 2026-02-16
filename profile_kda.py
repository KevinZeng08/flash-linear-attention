#!/usr/bin/env python3
"""
Profile script for KDA (Key-Dependent Attention) chunk operations.
Measures execution latency of chunk_kda_fwd and chunk_kda_bwd using torch profiler.
"""

import argparse
import random
import time

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function, schedule, tensorboard_trace_handler

from fla.ops.kda import chunk_kda

# Default real-world trace: 18 sequences, total length 8192
DEFAULT_REAL_WORLD_TRACE = [0, 247, 699, 982, 1688, 1985, 2383, 3081, 3526, 3973,
                            4096, 4824, 5101, 5919, 6426, 7137, 7392, 7800, 8192]

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
    min_seq_len: int, 
    variance: float = 1.0,
    seed: int = 42
) -> list:
    """
    生成随机的序列长度列表，满足：
    - 序列数量为 num_seqs
    - 总长度为 total_len
    - 每个序列长度 >= min_seq_len
    - variance: 方差控制参数
        - 0.0: 完全均衡，所有序列长度尽可能相等
        - 1.0: 正常随机分配
        - >1.0: 更不均衡，序列长度差异更大
    """
    assert total_len >= num_seqs * min_seq_len, \
        f"total_len ({total_len}) must be >= num_seqs ({num_seqs}) * min_seq_len ({min_seq_len})"
    
    # 计算均衡情况下每个序列的长度
    base_len = total_len // num_seqs
    remainder = total_len % num_seqs
    
    set_seed(seed)

    if variance == 0.0:
        # 完全均衡分配
        seq_lens = [base_len] * num_seqs
        # 将余数分配给前几个序列
        for i in range(remainder):
            seq_lens[i] += 1
    else:
        # 先给每个序列分配最小长度
        seq_lens = [min_seq_len] * num_seqs
        remaining = total_len - num_seqs * min_seq_len
        
        if remaining > 0:
            if variance >= 1.0:
                # 高方差：使用 Dirichlet 分布生成权重
                # alpha 越小，分布越不均匀
                alpha = 1.0 / variance
                weights = [random.gammavariate(alpha, 1.0) for _ in range(num_seqs)]
                total_weight = sum(weights)
                weights = [w / total_weight for w in weights]
                
                # 按权重分配剩余长度
                extra_lens = [int(remaining * w) for w in weights]
                # 处理舍入误差
                diff = remaining - sum(extra_lens)
                for i in range(abs(diff)):
                    idx = random.randint(0, num_seqs - 1)
                    extra_lens[idx] += 1 if diff > 0 else -1
                
                for i in range(num_seqs):
                    seq_lens[i] += extra_lens[i]
            else:
                # 低方差 (0 < variance < 1)：在均衡和随机之间插值
                # 先计算均衡分配
                balanced = [base_len] * num_seqs
                for i in range(remainder):
                    balanced[i] += 1
                
                # 计算随机分配
                random_lens = [min_seq_len] * num_seqs
                for _ in range(remaining):
                    idx = random.randint(0, num_seqs - 1)
                    random_lens[idx] += 1
                
                # 按 variance 插值
                seq_lens = [
                    int(balanced[i] * (1 - variance) + random_lens[i] * variance)
                    for i in range(num_seqs)
                ]
                # 修正总长度
                diff = total_len - sum(seq_lens)
                for i in range(abs(diff)):
                    idx = i % num_seqs
                    seq_lens[idx] += 1 if diff > 0 else -1
    
    # 确保所有序列长度 >= min_seq_len
    for i in range(num_seqs):
        if seq_lens[i] < min_seq_len:
            deficit = min_seq_len - seq_lens[i]
            seq_lens[i] = min_seq_len
            # 从其他序列借用
            for j in range(num_seqs):
                if j != i and seq_lens[j] > min_seq_len:
                    take = min(deficit, seq_lens[j] - min_seq_len)
                    seq_lens[j] -= take
                    deficit -= take
                    if deficit == 0:
                        break
    
    assert sum(seq_lens) == total_len, f"sum(seq_lens)={sum(seq_lens)} != total_len={total_len}"
    assert all(s >= min_seq_len for s in seq_lens), f"Some seq_len < min_seq_len"
    
    return seq_lens


def warmup(device, num_iters=10):
    """Warmup GPU to ensure stable measurements."""
    for _ in range(num_iters):
        a = torch.randn(1024, 1024, device=device)
        b = torch.randn(1024, 1024, device=device)
        torch.matmul(a, b)
    torch.cuda.synchronize() if device.type == 'cuda' else None


def create_input_data(B, T, H, K, V, device, dtype=torch.bfloat16, use_gate_in_kernel=True):
    """Create input tensors for KDA."""
    torch.manual_seed(42)

    q = torch.randn(B, T, H, K, dtype=dtype, device=device)
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    v = torch.randn(B, T, H, V, dtype=dtype, device=device)
    beta = torch.rand(B, T, H, dtype=dtype, device=device)
    g = torch.rand(B, T, H, K, dtype=dtype, device=device)
    do = torch.randn(B, T, H, K, dtype=dtype, device=device)

    A_log = torch.randn(H, dtype=torch.float32, device=device)
    dt_bias = torch.randn(H * K, dtype=torch.float32, device=device) if use_gate_in_kernel else None

    # Require gradients for backward pass
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    g.requires_grad_(True)
    beta.requires_grad_(True)

    return q, k, v, g, beta, A_log, dt_bias, do


def profile_kda_fwd(
    B, T, H, K, V,
    device='cuda',
    dtype=torch.bfloat16,
    use_gate_in_kernel=True,
    use_qk_l2norm_in_kernel=True,
    varlen=False,
    safe_gate=False,
    num_warmup=10,
    num_iters=20,
    with_profiler=False,
    profiler_sort_by='cuda_time_total',
):
    """Profile KDA forward pass."""
    print(f"\n{'='*60}")
    print(f"Profiling KDA Forward: B={B}, T={T}, H={H}, K={K}, V={V}")
    print(f"safe_gate={safe_gate}, use_gate_in_kernel={use_gate_in_kernel}")
    print(f"{'='*60}")

    device = torch.device(device)

    # Create input data
    if varlen:
        seqlens = generate_random_seq_lens(num_seqs=B, total_len=T, min_seq_len=63, variance=10.0, seed=42)
        print(seqlens)
        cum_seqlens = exclusive_cumsum(seqlens)
        cu_seqlens = torch.tensor(cum_seqlens, dtype=torch.int64, device=device)
        B = 1 # set B to 1
    else:
        cu_seqlens = None
    q, k, v, g, beta, A_log, dt_bias, do = create_input_data(
        B, T, H, K, V, device, dtype, use_gate_in_kernel
    )

    lower_bound = -5.0 if safe_gate else None

    # Warmup
    print(f"Warming up ({num_warmup} iterations)...")
    warmup(device, num_warmup)
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = chunk_kda(
                q, k, v, g, beta,
                A_log=A_log,
                dt_bias=dt_bias,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                use_gate_in_kernel=use_gate_in_kernel,
                cu_seqlens=cu_seqlens,
                safe_gate=safe_gate,
                lower_bound=lower_bound,
            )
    torch.cuda.synchronize() if device.type == 'cuda' else None

    # Profile with torch.profiler
    if with_profiler:
        print(f"\nRunning profiler ({num_iters} iterations)...")
        activities = [ProfilerActivity.CPU]
        if device.type == 'cuda':
            activities.append(ProfilerActivity.CUDA)

        with profile(
            activities=activities,
            schedule=schedule(wait=1, warmup=2, active=num_iters),
            on_trace_ready=tensorboard_trace_handler("./log/kda_fwd_profile"),
            record_shapes=True,
            with_stack=True,
        ) as prof:
            for _ in range(num_iters + 3):
                with record_function("chunk_kda_fwd"):
                    with torch.no_grad():
                        o, _ = chunk_kda(
                            q, k, v, g, beta,
                            A_log=A_log,
                            dt_bias=dt_bias,
                            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                            use_gate_in_kernel=use_gate_in_kernel,
                            cu_seqlens=cu_seqlens,
                            safe_gate=safe_gate,
                            lower_bound=lower_bound,
                        )
                prof.step()

        torch.cuda.synchronize() if device.type == 'cuda' else None


    # Simple timing measurement
    print(f"\n--- Simple Timing Measurement ({num_iters} iterations) ---")
    times = []
    for _ in range(num_iters):
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start = time.perf_counter()
        with torch.no_grad():
            o, _ = chunk_kda(
                q, k, v, g, beta,
                A_log=A_log,
                dt_bias=dt_bias,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                use_gate_in_kernel=use_gate_in_kernel,
                cu_seqlens=cu_seqlens,
                safe_gate=safe_gate,
                lower_bound=lower_bound,
            )
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    print(f"Forward pass: Avg={avg_time:.3f}ms, Min={min_time:.3f}ms, Max={max_time:.3f}ms")

    return avg_time


def profile_kda_fwd_bwd(
    B, T, H, K, V,
    device='cuda',
    dtype=torch.bfloat16,
    use_gate_in_kernel=True,
    use_qk_l2norm_in_kernel=True,
    varlen=False,
    safe_gate=True,
    num_warmup=10,
    num_iters=20,
    with_profiler=False,
    profiler_sort_by='cuda_time_total',
):
    """Profile KDA both forward and backward passes together."""
    print(f"\n{'='*60}")
    print(f"Profiling KDA FWD+BWD: B={B}, T={T}, H={H}, K={K}, V={V}")
    print(f"safe_gate={safe_gate}, use_gate_in_kernel={use_gate_in_kernel}")
    print(f"{'='*60}")

    device = torch.device(device)

    # Create input data
    if varlen:
        seqlens = generate_random_seq_lens(num_seqs=B, total_len=T, min_seq_len=63, variance=1.0, seed=42)
        cum_seqlens = exclusive_cumsum(seqlens)
        cu_seqlens = torch.tensor(cum_seqlens, dtype=torch.int64, device=device)
        B = 1 # set B to 1
    else:
        cu_seqlens = None
    q, k, v, g, beta, A_log, dt_bias, do = create_input_data(
        B, T, H, K, V, device, dtype, use_gate_in_kernel
    )

    lower_bound = -5.0 if safe_gate else None

    # Warmup
    print(f"Warming up ({num_warmup} iterations)...")
    for _ in range(num_warmup):
        q_grad = q.clone().requires_grad_(True)
        k_grad = k.clone().requires_grad_(True)
        v_grad = v.clone().requires_grad_(True)
        g_grad = g.clone().requires_grad_(True)
        beta_grad = beta.clone().requires_grad_(True)

        o, _ = chunk_kda(
            q_grad, k_grad, v_grad, g_grad, beta_grad,
            A_log=A_log,
            dt_bias=dt_bias,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_gate_in_kernel=use_gate_in_kernel,
            cu_seqlens=cu_seqlens,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
        )
        o.backward(do)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    print(cu_seqlens)

    # Profile with torch.profiler
    if with_profiler:
        print(f"\nRunning profiler ({num_iters} iterations)...")
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

        with profile(
            activities=activities,
            schedule=schedule(wait=1, warmup=2, active=num_iters),
            with_stack=False,
        ) as prof:
            for _ in range(num_iters + 3):
                q_grad = q.clone().requires_grad_(True)
                k_grad = k.clone().requires_grad_(True)
                v_grad = v.clone().requires_grad_(True)
                g_grad = g.clone().requires_grad_(True)
                beta_grad = beta.clone().requires_grad_(True)

                with record_function("chunk_kda_fwd_bwd"):
                    o, _ = chunk_kda(
                        q_grad, k_grad, v_grad, g_grad, beta_grad,
                        A_log=A_log,
                        dt_bias=dt_bias,
                        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                        use_gate_in_kernel=use_gate_in_kernel,
                        cu_seqlens=cu_seqlens,
                        safe_gate=safe_gate,
                        lower_bound=lower_bound,
                    )
                    o.backward(do)
                prof.step()

        torch.cuda.synchronize() if device.type == 'cuda' else None

        # Export chrome trace
        print("\n--- Exporting Chrome trace ---")
        output_path = f"kda_fwdbwd_{'varlen_' if varlen else ''}profile_trace.json"
        prof.export_chrome_trace(output_path)
        print(f"Trace saved to: {output_path}")

        # Memory summary
        print("\n--- Memory Summary ---")
        if device.type == 'cuda':
            print(f"Allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
            print(f"Max allocated: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB")
            print(f"Reserved: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")

    # Simple timing measurement
    print(f"\n--- Simple Timing Measurement ({num_iters} iterations) ---")
    times = []
    fwd_times = []
    bwd_times = []

    for _ in range(num_iters):
        q_grad = q.clone().requires_grad_(True)
        k_grad = k.clone().requires_grad_(True)
        v_grad = v.clone().requires_grad_(True)
        g_grad = g.clone().requires_grad_(True)
        beta_grad = beta.clone().requires_grad_(True)

        torch.cuda.synchronize() if device.type == 'cuda' else None
        start = time.perf_counter()
        o, _ = chunk_kda(
            q_grad, k_grad, v_grad, g_grad, beta_grad,
            A_log=A_log,
            dt_bias=dt_bias,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_gate_in_kernel=use_gate_in_kernel,
            cu_seqlens=cu_seqlens,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
        )
        torch.cuda.synchronize() if device.type == 'cuda' else None
        fwd_end = time.perf_counter()

        o.backward(do)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        bwd_end = time.perf_counter()

        fwd_times.append((fwd_end - start) * 1000)
        bwd_times.append((bwd_end - fwd_end) * 1000)
        times.append((bwd_end - start) * 1000)

    avg_time = sum(times) / len(times)
    avg_fwd = sum(fwd_times) / len(fwd_times)
    avg_bwd = sum(bwd_times) / len(bwd_times)

    print(f"Forward: Avg={avg_fwd:.3f}ms")
    print(f"Backward: Avg={avg_bwd:.3f}ms")
    print(f"Total: Avg={avg_time:.3f}ms, Min={min(times):.3f}ms, Max={max(times):.3f}ms")

    return avg_time, avg_fwd, avg_bwd


def sweep_configs(args):
    """Run profiling across multiple configurations."""
    configs = [
        # (B, T, H, K, V)
        (2, 2048, 4, 512, 512),
        (2, 4096, 4, 512, 512),
        (2, 8192, 4, 512, 512),
        (4, 2048, 4, 512, 512),
        (4, 4096, 4, 512, 512),
        (8, 2048, 4, 512, 512),
        (2, 2048, 8, 512, 512),
        (2, 2048, 4, 256, 512),
        (2, 2048, 4, 512, 256),
    ]

    results = []
    for B, T, H, K, V in configs:
        try:
            if args.mode in ['fwd', 'fwdbwd']:
                fwd_time = profile_kda_fwd(
                    B, T, H, K, V,
                    num_warmup=args.warmup,
                    num_iters=args.iters,
                    with_profiler=args.profiler and args.mode == 'fwd',
                    safe_gate=args.safe_gate,
                )

            if args.mode == 'fwdbwd':
                avg_time, avg_fwd, avg_bwd = profile_kda_fwd_bwd(
                    B, T, H, K, V,
                    num_warmup=args.warmup,
                    num_iters=args.iters,
                    with_profiler=args.profiler,
                    safe_gate=args.safe_gate,
                )

            results.append((B, T, H, K, V, locals().get('fwd_time', 0), locals().get('bwd_time', 0)))
        except Exception as e:
            print(f"Error with config (B={B}, T={T}, H={H}, K={K}, V={V}): {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    print("\n" + "="*80)
    print("Summary:")
    print("="*80)
    print(f"{'B':>4} {'T':>6} {'H':>4} {'K':>4} {'V':>4} {'Fwd(ms)':>10} {'Bwd(ms)':>10}")
    print("-"*80)
    for B, T, H, K, V, fwd, bwd in results:
        print(f"{B:>4} {T:>6} {H:>4} {K:>4} {V:>4} {fwd:>10.3f} {bwd:>10.3f}")


def main():
    parser = argparse.ArgumentParser(description='Profile KDA operations')
    parser.add_argument('--mode', type=str, default='fwdbwd',
                       choices=['fwd', 'bwd', 'fwdbwd'],
                       help='Profiling mode: forward, backward, or both')
    parser.add_argument('--B', type=int, default=1, help='Batch size')
    parser.add_argument('--T', type=int, default=8192, help='Sequence length')
    parser.add_argument('--H', type=int, default=64, help='Number of heads')
    parser.add_argument('--K', type=int, default=128, help='Key dimension')
    parser.add_argument('--V', type=int, default=128, help='Value dimension')
    parser.add_argument('--varlen', action='store_true', help='Use variable-length sequences (only supports B=1)')
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup iterations')
    parser.add_argument('--iters', type=int, default=20, help='Number of profiling iterations')
    parser.add_argument('--profiler', action='store_true', help='Use torch.profiler')
    parser.add_argument('--sweep', action='store_true', help='Run sweep over multiple configs')
    parser.add_argument('--sort_by', type=str, default='cuda_time_total',
                       choices=['cuda_time_total', 'cpu_time_total', 'cuda_time_avg'],
                       help='Sort profiler output by this metric')

    args = parser.parse_args()

    # Create log directory
    import os
    os.makedirs('./log', exist_ok=True)

    if args.sweep:
        sweep_configs(args)
    else:
        if args.mode == 'fwd':
            profile_kda_fwd(
                args.B, args.T, args.H, args.K, args.V,
                varlen=args.varlen,
                num_warmup=args.warmup,
                num_iters=args.iters,
                with_profiler=args.profiler,
                profiler_sort_by=args.sort_by,
            )
        else:
            profile_kda_fwd_bwd(
                args.B, args.T, args.H, args.K, args.V,
                varlen=args.varlen,
                num_warmup=args.warmup,
                num_iters=args.iters,
                with_profiler=args.profiler,
                profiler_sort_by=args.sort_by,
            )


if __name__ == '__main__':
    main()
