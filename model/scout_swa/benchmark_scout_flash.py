import math
import time
import torch
import argparse

from scout_attention_triton_v2 import flash_attn_scout_auto


@torch.no_grad()
def time_op(fn, iters=100, warmup=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1e3 / iters


def scout_swa_eager_forward(q_bhtd, k_bhtd, v_bhtd, sel_win, scale, causal=True, drop_diagonal=True):
    B, H, T, D = q_bhtd.shape
    device = q_bhtd.device
    dtype  = q_bhtd.dtype

    if T >= sel_win:
        sel_pos = torch.arange(sel_win - 1, T, sel_win, device=device)
        k_sel = k_bhtd[:, :, sel_win-1::sel_win, :]
        v_sel = v_bhtd[:, :, sel_win-1::sel_win, :]
    else:
        sel_pos = torch.empty(0, device=device, dtype=torch.long)
        k_sel = k_bhtd.new_zeros(B, H, 0, D)
        v_sel = v_bhtd.new_zeros(B, H, 0, D)
    Tk = k_sel.shape[2]

    logits_self = torch.einsum("bhtd,bhtd->bht", q_bhtd.float(), k_bhtd.float()) * scale
    logits_grid = torch.matmul(q_bhtd.float(), k_sel.float().transpose(-1, -2)) * scale

    if Tk > 0:
        i = torch.arange(T, device=device).view(1, 1, T, 1)
        jpos = sel_pos.view(1, 1, 1, Tk)
        allow = torch.ones((1,1,T,Tk), dtype=torch.bool, device=device)
        if causal:       allow = allow & (i >= jpos)
        if drop_diagonal: allow = allow & (i != jpos)
        neg = torch.finfo(torch.float32).min
        logits_grid = torch.where(allow, logits_grid, logits_grid.new_full((), neg))

    logits = torch.cat([logits_self.unsqueeze(-1), logits_grid], dim=-1)
    attn   = torch.softmax(logits, dim=-1, dtype=torch.float32).to(dtype)
    w_self, w_grid = attn[..., :1], attn[..., 1:]

    out = w_grid.matmul(v_sel) + w_self * v_bhtd
    return out


def run_benchmark(B, H, T, D, sel_win, dtype_str, causal=True, drop_diag=True, device="cuda", iters=100):
    """Run benchmark with the specified configuration"""
    if dtype_str.lower() == "bf16":
        dtype = torch.bfloat16
    elif dtype_str.lower() == "fp16":
        dtype = torch.float16
    else:  # fp32
        dtype = torch.float32
        
    print(f"\n=== SCOUT vs Triton Benchmark ===")
    print(f"B={B} H={H} T={T} D={D} sel_win={sel_win} dtype={dtype_str} causal={causal} drop_diag={drop_diag}\n")

    q = torch.randn(B, H, T, D, device=device, dtype=dtype)
    k = torch.randn(B, H, T, D, device=device, dtype=dtype)
    v = torch.randn(B, H, T, D, device=device, dtype=dtype)
    scale = 1.0 / math.sqrt(D)

    ref_out = scout_swa_eager_forward(q, k, v, sel_win=sel_win, scale=scale, causal=causal, drop_diagonal=drop_diag)

    # Triton expects (B,T,H,D)
    q_btHD = q.permute(0, 2, 1, 3).contiguous()
    k_btHD = k.permute(0, 2, 1, 3).contiguous()
    v_btHD = v.permute(0, 2, 1, 3).contiguous()
    if T >= sel_win:
        ksel_btHD = k_btHD[:, sel_win-1::sel_win, :, :].contiguous()
        vsel_btHD = v_btHD[:, sel_win-1::sel_win, :, :].contiguous()
        sel_positions = torch.arange(sel_win - 1, T, sel_win, device=device, dtype=torch.int32)
    else:
        ksel_btHD = k_btHD[:, 0:0, :, :]
        vsel_btHD = v_btHD[:, 0:0, :, :]
        sel_positions = torch.empty(0, device=device, dtype=torch.int32)

    kself_btHD = k_btHD
    vself_btHD = v_btHD

    # Test both layouts with flash_attn_scout_auto
    # Test (B,T,H,D) layout
    tri_out_btHD = flash_attn_scout_auto(
        q_btHD, ksel_btHD, vsel_btHD, kself_btHD, vself_btHD,
        sel_positions, causal=causal, softmax_scale=scale, drop_diagonal=drop_diag,
    )
    tri_out_btHD_to_bhtd = tri_out_btHD.permute(0, 2, 1, 3).contiguous()
    
    # Test (B,H,T,D) layout directly
    if T >= sel_win:
        k_sel_bhtd = k[:, :, sel_win-1::sel_win, :]
        v_sel_bhtd = v[:, :, sel_win-1::sel_win, :]
    else:
        k_sel_bhtd = k.new_zeros(B, H, 0, D)
        v_sel_bhtd = v.new_zeros(B, H, 0, D)
    
    tri_out_bhtd = flash_attn_scout_auto(
        q, k_sel_bhtd, v_sel_bhtd, k, v,
        sel_positions, causal=causal, softmax_scale=scale, drop_diagonal=drop_diag,
    )
    
    # Verify both layouts produce the same results
    layout_diff = (tri_out_btHD_to_bhtd - tri_out_bhtd).abs().max().item()
    print(f"Layout consistency check (BTHD vs BHTD): max diff = {layout_diff:.6e}")
    
    if layout_diff > 1e-4:
        print("WARNING: Different layouts produce different results!")
    else:
        print("✓ Both layouts produce consistent results")
    
    # Use the (B,H,T,D) result for comparison
    tri_out = tri_out_bhtd

    diff = (tri_out - ref_out).float()
    max_abs  = diff.abs().max().item()
    mean_abs = diff.abs().mean().item()
    mean_rel = (diff.abs() / (ref_out.float().abs() + 1e-6)).mean().item()

    print("Output error:")
    print(f"  max|Δ|  = {max_abs:.6e}")
    print(f"  mean|Δ| = {mean_abs:.6e}")
    print(f"  mean rel= {mean_rel:.6e}\n")

    # Find positions with largest differences
    flat_diff = diff.abs().view(-1)
    largest_indices = torch.topk(flat_diff, k=min(5, flat_diff.numel()))
    
    # Convert flat indices back to multi-dimensional indices
    largest_pos = []
    for idx in largest_indices.indices:
        # Convert flat index to multi-dimensional index
        idx_item = idx.item()
        d = idx_item % D
        idx_item //= D
        t = idx_item % T
        idx_item //= T
        h = idx_item % H
        b = idx_item // H
        largest_pos.append((b, h, t, d))
    
    # Print largest differences
    if max_abs > 1e-3:  # Only show details for significant differences
        print("Largest differences at positions (batch, head, seq_pos, dim):")
        for i, (b, h, t, d) in enumerate(largest_pos):
            torch_val = ref_out[b, h, t, d].item()
            triton_val = tri_out[b, h, t, d].item()
            diff_val = diff[b, h, t, d].item()
            print(f"  {i+1}. Pos ({b},{h},{t},{d}): PyTorch={torch_val:.6f}, Triton={triton_val:.6f}, Diff={diff_val:.6f}")
    
    # Timing
    def eager_call(): scout_swa_eager_forward(q, k, v, sel_win=sel_win, scale=scale, causal=causal, drop_diagonal=drop_diag)
    
    def triton_bthd_call():
        # Test (B,T,H,D) layout with reordering
        q_btHD = q.permute(0, 2, 1, 3).contiguous()
        k_btHD = k.permute(0, 2, 1, 3).contiguous()
        v_btHD = v.permute(0, 2, 1, 3).contiguous()
        if T >= sel_win:
            ksel_btHD = k_btHD[:, sel_win-1::sel_win, :, :].contiguous()
            vsel_btHD = v_btHD[:, sel_win-1::sel_win, :, :].contiguous()
            sel_positions = torch.arange(sel_win - 1, T, sel_win, device=device, dtype=torch.int32)
        else:
            ksel_btHD = k_btHD[:, 0:0, :, :]
            vsel_btHD = v_btHD[:, 0:0, :, :]
            sel_positions = torch.empty(0, device=device, dtype=torch.int32)
        kself_btHD = k_btHD
        vself_btHD = v_btHD
        
        tri_out_btHD = flash_attn_scout_auto(q_btHD, ksel_btHD, vsel_btHD, kself_btHD, vself_btHD, sel_positions,
                         causal=causal, softmax_scale=scale, drop_diagonal=drop_diag)
        # Include the output reordering back to BHTD
        tri_out = tri_out_btHD.permute(0, 2, 1, 3).contiguous()
    
    def triton_bhtd_call():
        # Test (B,H,T,D) layout directly (no reordering needed)
        if T >= sel_win:
            k_sel_bhtd = k[:, :, sel_win-1::sel_win, :]
            v_sel_bhtd = v[:, :, sel_win-1::sel_win, :]
            sel_positions = torch.arange(sel_win - 1, T, sel_win, device=device, dtype=torch.int32)
        else:
            k_sel_bhtd = k.new_zeros(B, H, 0, D)
            v_sel_bhtd = v.new_zeros(B, H, 0, D)
            sel_positions = torch.empty(0, device=device, dtype=torch.int32)
        
        tri_out = flash_attn_scout_auto(q, k_sel_bhtd, v_sel_bhtd, k, v, sel_positions,
                         causal=causal, softmax_scale=scale, drop_diagonal=drop_diag)

    t_ref = time_op(eager_call, iters=iters, warmup=10)
    t_tri_bthd = time_op(triton_bthd_call, iters=iters, warmup=10)
    t_tri_bhtd = time_op(triton_bhtd_call, iters=iters, warmup=10)

    print(f"Latency (avg over {iters} iters):")
    print(f"  eager_ref     : {t_ref:.3f} ms/iter")
    print(f"  triton (BTHD) : {t_tri_bthd:.3f} ms/iter")
    print(f"  triton (BHTD) : {t_tri_bhtd:.3f} ms/iter")
    print(f"  speedup (BTHD): {t_ref / max(t_tri_bthd, 1e-9):.2f}×")
    print(f"  speedup (BHTD): {t_ref / max(t_tri_bhtd, 1e-9):.2f}×")
    print(f"  BHTD vs BTHD  : {t_tri_bthd / max(t_tri_bhtd, 1e-9):.2f}×")
    
    # Return metrics for comparison across configurations
    return {
        'max_diff': max_abs,
        'mean_diff': mean_abs,
        'rel_diff': mean_rel,
        'torch_time': t_ref,
        'triton_bthd_time': t_tri_bthd,
        'triton_bhtd_time': t_tri_bhtd,
        'speedup_bthd': t_ref / max(t_tri_bthd, 1e-9),
        'speedup_bhtd': t_ref / max(t_tri_bhtd, 1e-9),
        'layout_ratio': t_tri_bthd / max(t_tri_bhtd, 1e-9)
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark SCOUT Sparse-Window Attention: PyTorch vs Triton")
    parser.add_argument("--B", type=int, default=2, help="Batch size")
    parser.add_argument("--H", type=int, default=32, help="Number of heads")
    parser.add_argument("--T", type=int, default=1024, help="Sequence length")
    parser.add_argument("--D", type=int, default=128, help="Head dimension")
    parser.add_argument("--sel", type=int, default=8, help="Selection window size")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"], help="Data type")
    parser.add_argument("--iters", type=int, default=100, help="Number of iterations for timing")
    parser.add_argument("--multi", action="store_true", help="Run multiple configurations")
    args = parser.parse_args()
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    
    # Run a single configuration if --multi is not set
    if not args.multi:
        run_benchmark(
            B=args.B, H=args.H, T=args.T, D=args.D, 
            sel_win=args.sel, dtype_str=args.dtype,
            iters=args.iters
        )
        return

    # Run multiple configurations for comprehensive testing
    print("=== SCOUT Benchmark: Comprehensive Testing ===")
    
    # Store results for comparison
    results = {}
    
    # Test different sequence lengths
    seq_lengths = [256, 512, 1024, 2048, 4096] 
    print("\n--- Testing different sequence lengths ---")
    for T in seq_lengths:
        key = f"T={T}"
        results[key] = run_benchmark(
            B=args.B, H=args.H, T=T, D=args.D, 
            sel_win=args.sel, dtype_str=args.dtype,
            iters=50  # Fewer iterations for multiple configs
        )
        
    # Test different batch sizes
    batch_sizes = [1, 2, 4, 8]
    print("\n--- Testing different batch sizes ---")
    for B in batch_sizes:
        key = f"B={B}"
        results[key] = run_benchmark(
            B=B, H=args.H, T=1024, D=args.D, 
            sel_win=args.sel, dtype_str=args.dtype,
            iters=50
        )
        
    # Test different head counts
    head_counts = [8, 16, 32, 64]
    print("\n--- Testing different head counts ---")
    for H in head_counts:
        key = f"H={H}"
        results[key] = run_benchmark(
            B=args.B, H=H, T=1024, D=args.D, 
            sel_win=args.sel, dtype_str=args.dtype,
            iters=50
        )
    
    # Test different selection window sizes
    windows = [4, 8, 16, 32]
    print("\n--- Testing different selection window sizes ---")
    for win in windows:
        key = f"win={win}"
        results[key] = run_benchmark(
            B=args.B, H=args.H, T=1024, D=args.D, 
            sel_win=win, dtype_str=args.dtype,
            iters=50
        )
    
    # Test different precision modes
    dtypes = ["fp16", "bf16", "fp32"]
    print("\n--- Testing different precision modes ---")
    for dtype_str in dtypes:
        key = f"dtype={dtype_str}"
        results[key] = run_benchmark(
            B=args.B, H=args.H, T=1024, D=args.D, 
            sel_win=args.sel, dtype_str=dtype_str,
            iters=50
        )
    
    # Summary of all configurations
    print("\n=== Summary of All Configurations ===")
    print(f"{'Configuration':<15} {'Max Diff':<12} {'Mean Diff':<12} {'Rel Diff':<12} {'PyTorch (ms)':<12} {'Triton BTHD (ms)':<15} {'Triton BHTD (ms)':<15} {'Speedup BTHD':<12} {'Speedup BHTD':<12} {'Layout Ratio':<12}")
    print("-" * 120)
    
    for config, metrics in results.items():
        print(f"{config:<15} {metrics['max_diff']:<12.6e} {metrics['mean_diff']:<12.6e} {metrics['rel_diff']:<12.6e} {metrics['torch_time']:<12.3f} {metrics['triton_bthd_time']:<15.3f} {metrics['triton_bhtd_time']:<15.3f} {metrics['speedup_bthd']:<12.2f}× {metrics['speedup_bhtd']:<12.2f}× {metrics['layout_ratio']:<12.2f}×")
    
    # Find best and worst cases
    max_diff_config = max(results.items(), key=lambda x: x[1]['max_diff'])
    min_diff_config = min(results.items(), key=lambda x: x[1]['max_diff'])
    max_speedup_bthd_config = max(results.items(), key=lambda x: x[1]['speedup_bthd'])
    max_speedup_bhtd_config = max(results.items(), key=lambda x: x[1]['speedup_bhtd'])
    
    print("\n=== Notable Results ===")
    print(f"Best numerical accuracy: {min_diff_config[0]} with max diff = {min_diff_config[1]['max_diff']:.6e}")
    print(f"Worst numerical accuracy: {max_diff_config[0]} with max diff = {max_diff_config[1]['max_diff']:.6e}")
    print(f"Best speedup (BTHD): {max_speedup_bthd_config[0]} with {max_speedup_bthd_config[1]['speedup_bthd']:.2f}× speedup")
    print(f"Best speedup (BHTD): {max_speedup_bhtd_config[0]} with {max_speedup_bhtd_config[1]['speedup_bhtd']:.2f}× speedup")
    
    # Generate visualizable data for further analysis
    print("\n=== Data for Visualization (CSV format) ===")
    print("config,max_diff,mean_diff,rel_diff,torch_time,triton_bthd_time,triton_bhtd_time,speedup_bthd,speedup_bhtd,layout_ratio")
    for config, metrics in results.items():
        print(f"{config},{metrics['max_diff']},{metrics['mean_diff']},{metrics['rel_diff']},{metrics['torch_time']},{metrics['triton_bthd_time']},{metrics['triton_bhtd_time']},{metrics['speedup_bthd']},{metrics['speedup_bhtd']},{metrics['layout_ratio']}")

if __name__ == "__main__":
    main()
