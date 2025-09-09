# Portions adapted from FlashAttention (Dao et al.), BSD-3-Clause.
# Copyright (c) 2022–2025, Tri Dao and contributors.
# See third_party/flash-attn/LICENSE.
import math

import torch
import triton
import triton.language as tl


# Disabling autotune for now, set num_warps=4 if headdim=64 and num_warps=8 if headdim=128
# Do not use autotune together with heuristics for BLOCK_M/BLOCK_N
# as this creates conflicts. Let's keep the heuristics approach.
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _fwd_kernel_scout(
    Q, K, V,                 # downsampled KV (Tk)
    KS, VS,                  # self KV    (Tq)  ← NEW
    SelPositions,            # original positions of selected keys (Tk,) ← NEW
    Bias, SelfBias,          # bias for selected KV and for self diag (optional) ← NEW
    Out, Lse, TMP,
    softmax_scale,
    # strides (batch, head, row/col)
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_kSb, stride_kSh, stride_kSm,       # ← NEW
    stride_vSb, stride_vSh, stride_vSm,       # ← NEW
    stride_sel_pos,                           # ← NEW
    stride_bb, stride_bh, stride_bm,
    stride_sbb, stride_sbh,                   # SelfBias strides: (B, H, Tq) packed as (b, h) + row ← NEW
    stride_ob, stride_oh, stride_om,
    nheads, seqlen_q, seqlen_k, seqlen_q_rounded, headdim,
    CACHE_KEY_SEQLEN_Q, CACHE_KEY_SEQLEN_K,
    BIAS_TYPE: tl.constexpr, IS_CAUSAL: tl.constexpr,
    HAS_SELF_BIAS: tl.constexpr,              # ← NEW
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # program ids
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    # row/col/dim offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    # base pointers
    q_ptrs  = Q  + off_b*stride_qb  + off_h*stride_qh  + (offs_m[:, None]*stride_qm + offs_d[None, :])
    k_ptrs  = K  + off_b*stride_kb  + off_h*stride_kh  + (offs_n[:, None]*stride_kn + offs_d[None, :])
    v_ptrs  = V  + off_b*stride_vb  + off_h*stride_vh  + (offs_n[:, None]*stride_vn + offs_d[None, :])
    ks_ptrs = KS + off_b*stride_kSb + off_h*stride_kSh + (offs_m[:, None]*stride_kSm + offs_d[None, :])  # self
    vs_ptrs = VS + off_b*stride_vSb + off_h*stride_vSh + (offs_m[:, None]*stride_vSm + offs_d[None, :])  # self
    if BIAS_TYPE == "vector":
        b_ptrs = Bias + off_b*stride_bb + off_h*stride_bh + offs_n
    elif BIAS_TYPE == "matrix":
        b_ptrs = Bias + off_b*stride_bb + off_h*stride_bh + (offs_m[:, None]*stride_bm + offs_n[None, :])
    if HAS_SELF_BIAS:
        sb_ptrs = SelfBias + off_b*stride_sbb + off_h*stride_sbh + offs_m

    t_ptrs = TMP + off_hb * seqlen_q_rounded + offs_m
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i   = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

    # ---- load q rows ----
    if EVEN_M & EVEN_N:
        q = tl.load(q_ptrs) if EVEN_HEADDIM else tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
        else:
            q = tl.load(q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)

    # =========================
    #  (A) SELF ITEM INJECTION
    # =========================
    # load self k/v for these rows
    if EVEN_HEADDIM:
        ks = tl.load(ks_ptrs)
        vs = tl.load(vs_ptrs)
    else:
        ks = tl.load(ks_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
        vs = tl.load(vs_ptrs, mask=offs_d[None, :] < headdim, other=0.0)

    # s_self = <q, ks> (rowwise)
    # Using dot by elementwise multiply + sum to avoid trans_b
    # Convert to float32 before sum for better precision
    s_self = tl.sum(q.to(tl.float32) * ks.to(tl.float32), axis=1)

    if BIAS_TYPE == "none":
        s_self_scaled = s_self * softmax_scale
    else:
        s_self_scaled = s_self * softmax_scale
        if HAS_SELF_BIAS:
            sb = tl.load(sb_ptrs).to(tl.float32)      # (BLOCK_M,)
            s_self_scaled = s_self_scaled + sb        # add diag bias

    # online-softmax update with a single element
    m_ij = tl.maximum(s_self_scaled, lse_i)
    p_self = tl.exp(s_self_scaled - m_ij)

    acc_o_scale = tl.exp(m_i - m_ij)
    tl.store(t_ptrs, acc_o_scale); acc_o_scale = tl.load(t_ptrs)
    acc_o = acc_o * acc_o_scale[:, None]
    acc_o += (p_self[:, None] * vs)

    m_i    = m_ij
    l_i_new = tl.exp(lse_i - m_ij) + p_self
    lse_i  = m_ij + tl.log(l_i_new)

    # =========================
    #  (B) GRID (DOWN-SAMPLED) KV
    # =========================
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1)*BLOCK_M, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # load K/V block
        if EVEN_N & EVEN_M:
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn)
                v = tl.load(v_ptrs + start_n * stride_vn)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_d[None, :] < headdim, other=0.0)
                v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=(start_n + offs_n)[:, None] < seqlen_k, other=0.0)
                v = tl.load(v_ptrs + start_n * stride_vn, mask=(start_n + offs_n)[:, None] < seqlen_k, other=0.0)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn,
                            mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                            other=0.0)
                v = tl.load(v_ptrs + start_n * stride_vn,
                            mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                            other=0.0)

        # For attention: q @ k.T where q is (BLOCK_M, BLOCK_HEADDIM) and k is (BLOCK_N, BLOCK_HEADDIM)
        # Need to transpose k to get (BLOCK_HEADDIM, BLOCK_N)
        k_t = tl.trans(k)  # (BLOCK_N, BLOCK_HEADDIM) -> (BLOCK_HEADDIM, BLOCK_N)
        qk = tl.dot(q, k_t)  # (BLOCK_M, BLOCK_HEADDIM) @ (BLOCK_HEADDIM, BLOCK_N) = (BLOCK_M, BLOCK_N)

        # Load actual selected positions for this block
        sel_pos_ptrs = SelPositions + start_n * stride_sel_pos + offs_n * stride_sel_pos
        sel_positions = tl.load(sel_pos_ptrs, mask=(start_n + offs_n) < seqlen_k, other=seqlen_q)
        
        # bounds & causal using actual selected positions
        if not EVEN_N:
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))
        if IS_CAUSAL:
            # Causal constraint: query position >= selected key position
            # We need to check if the actual query position is >= the position in the selected indices
            qk += tl.where(offs_m[:, None] >= sel_positions[None, :], 0, float("-inf"))
        
        # Remove diagonal to avoid double-counting self in grid branch
        # Query indices = offs_m, Selected key positions = sel_positions
        diag_mask = offs_m[:, None] == sel_positions[None, :]
        qk += tl.where(diag_mask, float("-inf"), 0)

        # bias (for selected keys only)
        if BIAS_TYPE != "none":
            if BIAS_TYPE == "vector":
                bias = tl.load(b_ptrs + start_n, mask=True if EVEN_N else (start_n + offs_n) < seqlen_k, other=0.0)
                qk = qk * softmax_scale + bias[None, :].to(tl.float32)
                m_ij = tl.maximum(tl.max(qk, 1), lse_i)
                p = tl.exp(qk - m_ij[:, None])
            else:  # "matrix"
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs + start_n)
                else:
                    bias = tl.load(b_ptrs + start_n,
                                   mask=(offs_m[:, None] < seqlen_q) & ((start_n + offs_n)[None, :] < seqlen_k),
                                   other=0.0)
                qk = qk * softmax_scale + bias.to(tl.float32)
                m_ij = tl.maximum(tl.max(qk, 1), lse_i)
                p = tl.exp(qk - m_ij[:, None])
        else:
            m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
            p = tl.exp(qk * softmax_scale - m_ij[:, None])

        l_ij = tl.sum(p, 1)
        acc_o_scale = tl.exp(m_i - m_ij)
        tl.store(t_ptrs, acc_o_scale); acc_o_scale = tl.load(t_ptrs)
        acc_o = acc_o * acc_o_scale[:, None]
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)

        m_i   = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)

    # finalize
    o_scale = tl.exp(m_i - lse_i)
    tl.store(t_ptrs, o_scale); o_scale = tl.load(t_ptrs)
    acc_o = acc_o * o_scale[:, None]

    # write back
    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m
    tl.store(lse_ptrs, lse_i)

    outs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = Out + off_b*stride_ob + off_h*stride_oh + (offs_m[:, None]*stride_om + outs_d[None, :])
    if EVEN_M:
        tl.store(out_ptrs, acc_o if EVEN_HEADDIM else acc_o, mask=(outs_d[None, :] < headdim))
    else:
        tl.store(out_ptrs, acc_o, mask=(offs_m[:, None] < seqlen_q) & (outs_d[None, :] < headdim))



def _flash_attn_forward_scout(q, k_sel, v_sel, k_self, v_self, bias=None, self_bias=None, causal=False, softmax_scale=None, sel_positions=None):
    B, Tq, H, Dh = q.shape
    _, Tk, _, _  = k_sel.shape
    softmax_scale = softmax_scale or 1.0 / math.sqrt(Dh)
    
    # Create selected positions if not provided
    if sel_positions is None:
        # Assume uniform selection starting at sel_win-1
        # This is a fallback - should be provided explicitly
        sel_win = Tq // Tk if Tk > 0 else 1
        # Make sure positions are strictly increasing and within bounds
        sel_positions = torch.arange(sel_win - 1, Tq, sel_win, device=q.device, dtype=torch.int32)
        sel_positions = sel_positions[sel_positions < Tq]
    else:
        sel_positions = sel_positions.to(device=q.device, dtype=torch.int32)

    has_bias = bias is not None
    bias_type = "none"
    if has_bias:
        if bias.stride(-1) != 1:
            bias = bias.contiguous()
        # bias is for selected keys only
        if bias.shape[2:] == (1, Tk):     bias_type = "vector"
        elif bias.shape[2:] == (Tq, Tk):  bias_type = "matrix"
        else: raise RuntimeError("bias last 2 dims must be (1,Tk) or (Tq,Tk)")
        bias = bias.expand(B, H, Tq, Tk)

    has_self_bias = self_bias is not None
    if has_self_bias:
        assert self_bias.shape == (B, H, Tq)
        if self_bias.stride(-1) != 1:
            self_bias = self_bias.contiguous()

    Tq_round = math.ceil(Tq / 128) * 128
    lse = torch.empty((B, H, Tq_round), device=q.device, dtype=torch.float32)
    tmp = torch.empty((B, H, Tq_round), device=q.device, dtype=torch.float32)
    o = torch.empty_like(q)

    BLOCK_HEADDIM = max(triton.next_power_of_2(Dh), 16)
    BLOCK = min(128, max(16, triton.next_power_of_2(Tq)))
    grid = lambda META: (triton.cdiv(Tq, META["BLOCK_M"]), B * H)
    _fwd_kernel_scout[grid](
        q, k_sel, v_sel, k_self, v_self, sel_positions,
        bias if has_bias else torch.empty(1, device=q.device),
        self_bias if has_self_bias else torch.empty(1, device=q.device),
        o, lse, tmp, softmax_scale,
        q.stride(0), q.stride(2), q.stride(1),
        k_sel.stride(0), k_sel.stride(2), k_sel.stride(1),
        v_sel.stride(0), v_sel.stride(2), v_sel.stride(1),
        k_self.stride(0), k_self.stride(2), k_self.stride(1),
        v_self.stride(0), v_self.stride(2), v_self.stride(1),
        sel_positions.stride(0),
        *(bias.stride(0), bias.stride(1), bias.stride(2)) if has_bias else (0, 0, 0),
        (self_bias.stride(0) if has_self_bias else 0),
        (self_bias.stride(1) if has_self_bias else 0),
        o.stride(0), o.stride(2), o.stride(1),
        H, Tq, Tk, Tq_round, Dh,
        Tq // 32, Tk // 32,
        bias_type, causal,
        HAS_SELF_BIAS=has_self_bias,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        BLOCK_M=BLOCK, BLOCK_N=BLOCK,
        # Use more warps for larger head dimensions for better parallelism
    num_warps=(4 if Dh <= 64 else 8), 
    # Keep stages at 1 for now until optimizing for memory hierarchy
    num_stages=1,
    )
    return o, lse, softmax_scale


@triton.jit
def _bwd_store_dk_dv(ptrs, _, dk_dv, dk_dv_copy, offs_n, offs_d, seqlen_k, headdim,
                     EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_HEADDIM: tl.constexpr):
    """Helper function to store dk/dv with proper masking"""
    if EVEN_N & EVEN_HEADDIM:
        tl.store(ptrs, dk_dv)
    else:
        tl.store(ptrs, dk_dv, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))


@triton.jit
def _bwd_kernel_scout_one_col_block(
    start_n,
    Q, K, V, KS, VS, Bias, SelfBias,
    DO, DQ, DK, DV, DKS, DVS, LSE, Delta,
    softmax_scale,
    stride_qm, stride_kn, stride_vn,
    stride_kSm, stride_vSm,          # self strides over row (Tq)
    stride_bm, stride_sbm,           # bias strides (matrix / self vector)
    stride_dom, stride_dqm, stride_dkn, stride_dvn, stride_dkSm, stride_dvSm,
    seqlen_q, seqlen_k, headdim,
    ATOMIC_ADD: tl.constexpr, BIAS_TYPE: tl.constexpr, IS_CAUSAL: tl.constexpr,
    HAS_SELF_BIAS: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    begin_m = 0 if not IS_CAUSAL else ((start_n * BLOCK_N) // BLOCK_M) * BLOCK_M
    offs_qm = begin_m + tl.arange(0, BLOCK_M)
    offs_n  = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m  = tl.arange(0, BLOCK_M)
    offs_d  = tl.arange(0, BLOCK_HEADDIM)

    q_ptrs   = Q   + (offs_qm[:, None]*stride_qm  + offs_d[None, :])
    k_ptrs   = K   + (offs_n[:, None]*stride_kn   + offs_d[None, :])
    v_ptrs   = V   + (offs_n[:, None]*stride_vn   + offs_d[None, :])
    ks_ptrs  = KS  + (offs_qm[:, None]*stride_kSm + offs_d[None, :])
    vs_ptrs  = VS  + (offs_qm[:, None]*stride_vSm + offs_d[None, :])
    do_ptrs  = DO  + (offs_qm[:, None]*stride_dom + offs_d[None, :])
    dq_ptrs  = DQ  + (offs_qm[:, None]*stride_dqm + offs_d[None, :])
    dks_ptrs = DKS + (offs_qm[:, None]*stride_dkSm + offs_d[None, :])
    dvs_ptrs = DVS + (offs_qm[:, None]*stride_dvSm + offs_d[None, :])

    # early exit if no rows in this block
    if begin_m >= seqlen_q:
        return

    # load row block of q / do
    if EVEN_M & EVEN_HEADDIM:
        q  = tl.load(q_ptrs)
        do = tl.load(do_ptrs)
    else:
        q  = tl.load(q_ptrs,  mask=(offs_qm[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)
        do = tl.load(do_ptrs, mask=(offs_qm[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)

    # -------------------------
    # Self item (per-row) grads
    # -------------------------
    if EVEN_HEADDIM:
        ks = tl.load(ks_ptrs);  vs = tl.load(vs_ptrs)
    else:
        ks = tl.load(ks_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
        vs = tl.load(vs_ptrs, mask=offs_d[None, :] < headdim, other=0.0)

    lse_i = tl.load(LSE + offs_qm)                 # (BLOCK_M,)
    s_self = tl.sum(q * ks, axis=1).to(tl.float32)
    if BIAS_TYPE == "none":
        s_self_scaled = s_self * softmax_scale
    else:
        s_self_scaled = s_self * softmax_scale
        if HAS_SELF_BIAS:
            sb = tl.load(SelfBias + offs_qm*stride_sbm).to(tl.float32)
            s_self_scaled = s_self_scaled + sb

    p_self = tl.exp(s_self_scaled - lse_i)         # (BLOCK_M,)

    # dv_self += p_self^T * do  (rowwise)
    dvs_add = (p_self[:, None].to(do.dtype)) * do
    tl.store(dvs_ptrs, dvs_add if EVEN_HEADDIM else dvs_add, mask=(offs_d[None, :] < headdim))

    # dp_self = do @ vs^T  (rowwise), ds_self = p_self * (dp_self - Delta) * scale
    dp_self = tl.sum(do * vs, axis=1).to(tl.float32)
    Di = tl.load(Delta + offs_qm)
    ds_self = (p_self * (dp_self - Di) * softmax_scale).to(q.dtype)

    # dk_self += ds_self^T @ q   ;  dq += ds_self @ ks
    dks_add = (ds_self[:, None].to(q.dtype)) * q
    tl.store(dks_ptrs, dks_add if EVEN_HEADDIM else dks_add, mask=(offs_d[None, :] < headdim))

    dq_add = (ds_self[:, None].to(q.dtype)) * ks
    # accumulate into DQ like base kernel
    if EVEN_M & EVEN_HEADDIM:
        dq_prev = tl.load(dq_ptrs, eviction_policy="evict_last")
        tl.store(dq_ptrs, dq_prev + dq_add, eviction_policy="evict_last")
    else:
        dq_prev = tl.load(dq_ptrs,
                          mask=(offs_qm[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                          other=0.0, eviction_policy="evict_last")
        tl.store(dq_ptrs, dq_prev + dq_add,
                 mask=(offs_qm[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                 eviction_policy="evict_last")

    # -------------------------
    # Selected KV columns grads (same as base)
    # -------------------------
    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs); v = tl.load(v_ptrs)
        else:
            k = tl.load(k_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            v = tl.load(v_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
            v = tl.load(v_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
        else:
            k = tl.load(k_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
            v = tl.load(v_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)

    # recompute p over selected KV  
    k_t = tl.trans(k)  # Transpose k for matrix multiplication
    qk = tl.dot(q, k_t)
    if not EVEN_N:  qk = tl.where(offs_n[None, :] < seqlen_k, qk, float("-inf"))
    if IS_CAUSAL:   
        # Use simple index-based causal masking for backward pass
        # This works when processing blocks sequentially
        qk = tl.where(offs_qm[:, None] >= offs_n[None, :], qk, float("-inf"))
    if BIAS_TYPE != "none":
        if BIAS_TYPE == "vector":
            # bias for selected KV only; vector handled in Python by expand
            pass
        else:  # matrix
            bias = tl.load(Bias + (offs_qm[:, None]*stride_bm + offs_n[None, :]),
                           mask=(offs_qm[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k),
                           other=0.0).to(tl.float32)
            qk = qk * softmax_scale + bias
            tl.debug_barrier()
    lse_i = tl.load(LSE + offs_qm)
    p = tl.exp((qk * softmax_scale) - lse_i[:, None])

    # dv (selected)
    p_t = tl.trans(p.to(do.dtype))  # Transpose p for matrix multiplication
    dv = tl.dot(p_t, do)
    dv_ptrs = DV + (offs_n[:, None]*stride_dvn + offs_d[None, :])
    _bwd_store_dk_dv(dv_ptrs, dv_ptrs, dv, dv, offs_n, offs_d, seqlen_k, headdim,
                     EVEN_M=EVEN_M, EVEN_N=EVEN_N, EVEN_HEADDIM=EVEN_HEADDIM)  # store dv

    # dp, ds, dk
    v_t = tl.trans(v)  # Transpose v for matrix multiplication
    dp = tl.dot(do, v_t)
    Di = tl.load(Delta + offs_qm)
    ds = (p * (dp - Di[:, None]) * softmax_scale).to(q.dtype)
    ds_t = tl.trans(ds)  # Transpose ds for matrix multiplication
    dk = tl.dot(ds_t, q)
    dk_ptrs = DK + (offs_n[:, None]*stride_dkn + offs_d[None, :])
    _bwd_store_dk_dv(dk_ptrs, dk_ptrs, dk, dk, offs_n, offs_d, seqlen_k, headdim,
                     EVEN_M=EVEN_M, EVEN_N=EVEN_N, EVEN_HEADDIM=EVEN_HEADDIM)

    # dq add
    dq_blk = tl.dot(ds, k)
    if EVEN_M & EVEN_HEADDIM:
        dq_prev = tl.load(dq_ptrs, eviction_policy="evict_last")
        tl.store(dq_ptrs, dq_prev + dq_blk, eviction_policy="evict_last")
    else:
        dq_prev = tl.load(dq_ptrs, mask=(offs_qm[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                          other=0.0, eviction_policy="evict_last")
        tl.store(dq_ptrs, dq_prev + dq_blk,
                 mask=(offs_qm[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                 eviction_policy="evict_last")





def _compute_exact_gradients_pytorch(
    do, q, k_sel, v_sel, k_self, v_self, o, lse, delta,
    dq, dk_sel, dv_sel, dk_self, dv_self,
    bias, self_bias, causal, softmax_scale, sel_positions
):
    """
    Exact gradients for SCOUT attention in PyTorch, matching the forward:
      logits_self = scale * <q, k_self> + self_bias
      logits_grid = scale * (q @ k_sel^T) + bias
      p_self  = exp(logits_self - lse)
      p_grid  = exp(logits_grid - lse)
      out     = p_self * v_self + p_grid @ v_sel
    Backprop uses the softmax identity: d_logits = p * (dS - delta),
    with dS_self = <do, v_self>, dS_grid = do @ v_sel^T.
    """
    B, Tq, H, Dh = q.shape
    _, Tk, _, _  = k_sel.shape
    scale = softmax_scale or (1.0 / math.sqrt(Dh))

    # Move to (B, H, T, D) for clean batched matmuls; keep math in fp32
    qf      = q.to(torch.float32).permute(0, 2, 1, 3)      # (B,H,Tq,D)
    kself_f = k_self.to(torch.float32).permute(0, 2, 1, 3) # (B,H,Tq,D)
    vself_f = v_self.to(torch.float32).permute(0, 2, 1, 3) # (B,H,Tq,D)
    ksel_f  = k_sel.to(torch.float32).permute(0, 2, 1, 3)  # (B,H,Tk,D)
    vsel_f  = v_sel.to(torch.float32).permute(0, 2, 1, 3)  # (B,H,Tk,D)
    do_f    = do.to(torch.float32).permute(0, 2, 1, 3)     # (B,H,Tq,D)

    # LSE is stored as (B,H,Tq_round); use valid Tq
    lse_f = lse[..., :Tq].to(torch.float32)                # (B,H,Tq)
    delta_f = delta.permute(0, 2, 1).to(torch.float32)     # (B,H,Tq)

    # Build logits
    # Self
    logits_self = (qf * kself_f).sum(dim=-1) * scale       # (B,H,Tq)
    if self_bias is not None:
        logits_self = logits_self + self_bias.to(torch.float32)

    # Grid
    logits_grid = torch.matmul(qf, ksel_f.transpose(-1, -2)) * scale  # (B,H,Tq,Tk)
    if bias is not None:
        logits_grid = logits_grid + bias.to(torch.float32)            # (B,H,Tq,Tk)

    # Masks using sel_positions (same semantics as forward)
    if sel_positions is None:
        sel_win = max(1, Tq // max(Tk, 1))
        sel_positions = torch.arange(sel_win - 1, Tq, sel_win, device=q.device, dtype=torch.int64)[:Tk]
    sel_positions = sel_positions.to(q.device, dtype=torch.int64)  # (Tk,)
    valid_j = sel_positions < Tq                                    # (Tk,)
    i_idx   = torch.arange(Tq, device=q.device).view(1, 1, Tq, 1)   # (1,1,Tq,1)
    pos_j   = sel_positions.view(1, 1, 1, Tk)                       # (1,1,1,Tk)

    # Always drop diagonal in the grid branch to avoid double-counting the self item
    diag_mask = (i_idx == pos_j)
    # Causal: allow only j with pos_j <= i; Non-causal: allow all valid_j
    if causal:
        allow = (i_idx >= pos_j) & valid_j.view(1,1,1,Tk)
    else:
        allow = valid_j.view(1,1,1,Tk)
    # Remove diagonal always
    allow = allow & (~diag_mask)
    # Apply mask
    neg_inf = torch.finfo(logits_grid.dtype).min
    logits_grid = torch.where(allow, logits_grid, torch.full_like(logits_grid, neg_inf))

    # Probabilities relative to total LSE
    p_self = torch.exp(logits_self - lse_f)                       # (B,H,Tq)
    p_grid = torch.exp(logits_grid - lse_f.unsqueeze(-1))         # (B,H,Tq,Tk)

    # dS terms
    dp_self = (do_f * vself_f).sum(dim=-1)                        # (B,H,Tq)
    dp_grid = torch.matmul(do_f, vsel_f.transpose(-1, -2))        # (B,H,Tq,Tk)

    # d logits via softmax identity
    ds_self = p_self * (dp_self - delta_f) * scale                # (B,H,Tq)
    ds_grid = p_grid * (dp_grid - delta_f.unsqueeze(-1)) * scale  # (B,H,Tq,Tk)

    # Gradients
    # dv
    dvself = p_self.unsqueeze(-1) * do_f                           # (B,H,Tq,D)
    dvsel  = torch.matmul(p_grid.transpose(-1, -2), do_f)          # (B,H,Tk,D)

    # dk
    dkself = ds_self.unsqueeze(-1) * qf                            # (B,H,Tq,D)
    dksel  = torch.matmul(ds_grid.transpose(-1, -2), qf)           # (B,H,Tk,D)

    # dq
    dq_from_self = ds_self.unsqueeze(-1) * kself_f                 # (B,H,Tq,D)
    dq_from_sel  = torch.matmul(ds_grid, ksel_f)                   # (B,H,Tq,D)
    dq_full = dq_from_self + dq_from_sel                           # (B,H,Tq,D)

    # Write into provided buffers (back to (B,Tq,H,D), cast to original dtypes)
    dq.copy_(dq_full.permute(0, 2, 1, 3).to(dq.dtype))
    dk_self.copy_(dkself.permute(0, 2, 1, 3).to(dk_self.dtype))
    dv_self.copy_(dvself.permute(0, 2, 1, 3).to(dv_self.dtype))

    # Zero out invalid selected positions
    if Tk > 0:
        if not valid_j.all():
            mask_j = valid_j.to(dk_sel.dtype).view(1,1,Tk,1)       # (1,1,Tk,1)
            dksel  = dksel * mask_j
            dvsel  = dvsel * mask_j
        dk_sel.copy_(dksel.permute(0, 2, 1, 3).to(dk_sel.dtype))
        dv_sel.copy_(dvsel.permute(0, 2, 1, 3).to(dv_sel.dtype))


def _flash_attn_backward_scout(
    do, q, k_sel, v_sel, k_self, v_self, o, lse,
    dq, dk_sel, dv_sel, dk_self, dv_self,
    bias=None, self_bias=None, causal=False, softmax_scale=None, sel_positions=None
):
    """
    Exact backward dispatcher (PyTorch version). Alloc/strides handled by caller.
    """
    # Delta = sum(do * o) over head-dim (matches forward online-softmax)
    delta = torch.sum(do * o, dim=-1)  # (B,Tq,H)

    _compute_exact_gradients_pytorch(
        do, q, k_sel, v_sel, k_self, v_self, o, lse, delta,
        dq, dk_sel, dv_sel, dk_self, dv_self,
        bias, self_bias, causal, softmax_scale, sel_positions
    )


class FlashAttnScoutFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k_sel, v_sel, k_self, v_self, bias=None, self_bias=None, causal=False, softmax_scale=None, sel_positions=None):
        """
        q:       (B, Tq, H, Dh)
        k_sel:   (B, Tk, H, Dh)     # downsampled keys
        v_sel:   (B, Tk, H, Dh)     # downsampled values
        k_self:  (B, Tq, H, Dh)     # per-row (self) keys (aligned to q rows)
        v_self:  (B, Tq, H, Dh)     # per-row (self) values
        bias:    optional broadcastable to (B, H, Tq, Tk)   # bias for *selected* K
        self_bias: optional (B, H, Tq)                      # bias for the self item (diagonal)
        causal:  usual causal flag
        sel_positions: optional (Tk,) tensor with original positions of selected keys
        """
        q, k_sel, v_sel, k_self, v_self = [x if x.stride(-1) == 1 else x.contiguous()
                                           for x in [q, k_sel, v_sel, k_self, v_self]]
        o, lse, ctx.softmax_scale = _flash_attn_forward_scout(
            q, k_sel, v_sel, k_self, v_self, bias=bias, self_bias=self_bias,
            causal=causal, softmax_scale=softmax_scale, sel_positions=sel_positions
        )
        ctx.save_for_backward(q, k_sel, v_sel, k_self, v_self, o, lse)
        ctx.causal = causal
        ctx.sel_positions = sel_positions
        ctx.bias = bias
        ctx.self_bias = self_bias
        return o

    @staticmethod
    def backward(ctx, do):
        q, k_sel, v_sel, k_self, v_self, o, lse = ctx.saved_tensors
        
        # Create gradient tensors
        dq = torch.zeros_like(q)
        dk_sel = torch.zeros_like(k_sel)
        dv_sel = torch.zeros_like(v_sel)
        dk_self = torch.zeros_like(k_self)
        dv_self = torch.zeros_like(v_self)
        
        _flash_attn_backward_scout(
            do, q, k_sel, v_sel, k_self, v_self, o, lse,
            dq, dk_sel, dv_sel, dk_self, dv_self,
            bias=ctx.bias, self_bias=ctx.self_bias, causal=ctx.causal,
            softmax_scale=ctx.softmax_scale, sel_positions=ctx.sel_positions
        )
        
        return dq, dk_sel, dv_sel, dk_self, dv_self, None, None, None, None, None


def flash_attn_scout(q, k_sel, v_sel, k_self, v_self, bias=None, self_bias=None, causal=False, softmax_scale=None, sel_positions=None):
    """
    Wrapper function for FlashAttnScoutFunc that accepts keyword arguments.
    """
    return FlashAttnScoutFunc.apply(q, k_sel, v_sel, k_self, v_self, bias, self_bias, causal, softmax_scale, sel_positions)



