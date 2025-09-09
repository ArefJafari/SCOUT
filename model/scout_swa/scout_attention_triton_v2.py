# Forward-only Triton SCOUT attention that matches the eager ScoutSWA logic:
# single softmax over [ self-item | selected grid ], causal on absolute positions,
# optional drop-diagonal. Inputs use (B, T, H, D) to minimize transposes in the caller.

import math
import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_scout_exact(
    Q, KSEL, VSEL, KSELF, VSELF, SELPOS,  # pointers
    OUT, LSE,
    SCALE,                                 # scalar float
    # strides (for (B,T,H,D) layout)
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_kSb, stride_kSh, stride_kSm,
    stride_vSb, stride_vSh, stride_vSm,
    stride_sel,
    stride_ob, stride_oh, stride_om,
    # sizes
    NHEADS, SEQLEN_Q, SEQLEN_K, SEQLEN_Q_ROUND, HEADDIM,
    # compile-time
    IS_CAUSAL: tl.constexpr, DROP_DIAG: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """
    Q:      (B, H, Tq, D)
    KSEL/VSEL:  (B, H, Tk, D)  -- grid selected KV
    KSELF/VSELF:(B, H, Tq, D)  -- self item (aligned with query time)
    SELPOS: (Tk,) int32        -- absolute positions of selected keys in [0, Tq)
    OUT:    (B, H, Tq, D)
    LSE:    (B*H, Tq_round)    -- m + logsumexp per row (for debugging)
    """

    pid_m = tl.program_id(0)             # blocks along Tq
    pid_bh = tl.program_id(1)            # fused (B,H)
    b = pid_bh // NHEADS
    h = pid_bh % NHEADS

    row_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # (BLOCK_M,)
    d_offs   = tl.arange(0, BLOCK_D)                     # (BLOCK_D,)
    n_offs   = tl.arange(0, BLOCK_N)                     # (BLOCK_N,)

    # Pointers for this (b,h) block
    q_ptrs   = Q     + b*stride_qb  + h*stride_qh  + (row_offs[:, None]*stride_qm + d_offs[None, :])
    ks_ptrs  = KSELF + b*stride_kSb + h*stride_kSh + (row_offs[:, None]*stride_kSm + d_offs[None, :])
    vs_ptrs  = VSELF + b*stride_vSb + h*stride_vSh + (row_offs[:, None]*stride_vSm + d_offs[None, :])
    k_ptrs   = KSEL  + b*stride_kb  + h*stride_kh  + (n_offs[:, None]*stride_kn + d_offs[None, :])
    v_ptrs   = VSEL  + b*stride_vb  + h*stride_vh  + (n_offs[:, None]*stride_vn + d_offs[None, :])

    # Output/LSE pointers
    out_ptrs = OUT + b*stride_ob + h*stride_oh + (row_offs[:, None]*stride_om + d_offs[None, :])
    lse_ptrs = LSE + (b*NHEADS + h) * SEQLEN_Q_ROUND + row_offs

    # Masks
    row_mask = row_offs < SEQLEN_Q
    d_mask   = d_offs   < HEADDIM
    mask2d   = (row_mask[:, None]) & (d_mask[None, :])

    # ---- Load Q / self-KV (masked) ----
    q  = tl.load(q_ptrs,  mask=mask2d, other=0.0)
    ks = tl.load(ks_ptrs, mask=mask2d, other=0.0)
    vs = tl.load(vs_ptrs, mask=mask2d, other=0.0)

    # fp32 compute
    qf  = q.to(tl.float32)
    ksf = ks.to(tl.float32)
    vsf = vs.to(tl.float32)

    # Online softmax accumulators
    NEG_BIG = -1.0e9
    m_i = tl.full([BLOCK_M], NEG_BIG, tl.float32)     # running max per row
    l_i = tl.zeros([BLOCK_M], tl.float32)             # running exp-sum per row
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)    # running numerator

    scale = SCALE

    # ===== Self item =====
    s_self = tl.sum(qf * ksf, axis=1) * scale     # (BLOCK_M,)
    m_new  = tl.maximum(m_i, s_self)
    alpha  = tl.exp(m_i - m_new)
    p_self = tl.exp(s_self - m_new)
    l_i = alpha * l_i + p_self
    acc = alpha[:, None] * acc + p_self[:, None] * vsf
    m_i = m_new

    # ===== Selected grid =====
    end_n = SEQLEN_K
    i_abs = row_offs  # (BLOCK_M,)

    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        col_idx = start_n + n_offs                           # (BLOCK_N,)
        col_mask = col_idx < SEQLEN_K

        k_blk = tl.load(k_ptrs + start_n * stride_kn,
                        mask=(col_mask[:, None] & d_mask[None, :]),
                        other=0.0)
        v_blk = tl.load(v_ptrs + start_n * stride_vn,
                        mask=(col_mask[:, None] & d_mask[None, :]),
                        other=0.0)

        kf = k_blk.to(tl.float32)    # (BLOCK_N, D)
        vf = v_blk.to(tl.float32)    # (BLOCK_N, D)

        # qk = (M,D) x (D,N) -> (M,N)
        qk = tl.dot(qf, tl.trans(kf)) * scale

        # Load absolute positions of selected keys for these columns
        selpos_ptrs = SELPOS + col_idx * stride_sel
        selpos = tl.load(selpos_ptrs, mask=col_mask, other=0).to(tl.int32)   # (BLOCK_N,)

        # Build allow mask on absolute positions
        allow = col_mask[None, :]  # (M,N)
        if IS_CAUSAL:
            allow = allow & (i_abs[:, None] >= selpos[None, :])
        if DROP_DIAG:
            allow = allow & (i_abs[:, None] != selpos[None, :])

        # Mask by writing big negative
        qk = tl.where(allow, qk, NEG_BIG)

        # Online softmax update
        m_blk = tl.max(qk, axis=1)               # (M,)
        m_new = tl.maximum(m_i, m_blk)
        alpha = tl.exp(m_i - m_new)              # (M,)
        p_blk = tl.exp(qk - m_new[:, None])      # (M,N)

        l_i = alpha * l_i + tl.sum(p_blk, axis=1)
        acc = alpha[:, None] * acc + tl.dot(p_blk, vf)
        m_i = m_new

    # finalize
    inv_l = 1.0 / tl.maximum(l_i, 1e-20)
    o = acc * inv_l[:, None]

    # store
    tl.store(out_ptrs, o, mask=mask2d)
    tl.store(lse_ptrs, (m_i + tl.log(l_i)), mask=row_mask)


def flash_attn_scout(
    q_btHD: torch.Tensor,       # (B, Tq, H, D)
    ksel_btHD: torch.Tensor,    # (B, Tk, H, D)
    vsel_btHD: torch.Tensor,    # (B, Tk, H, D)
    kself_btHD: torch.Tensor,   # (B, Tq, H, D)
    vself_btHD: torch.Tensor,   # (B, Tq, H, D)
    sel_positions: torch.Tensor,  # (Tk,) int32 absolute positions
    causal: bool = True,
    softmax_scale: float | None = None,
    drop_diagonal: bool = True,
):
    """Forward-only SCOUT Triton kernel (returns (B, Tq, H, D))."""
    assert q_btHD.is_cuda, "CUDA tensor required"
    B, Tq, H, D = q_btHD.shape
    Tk = ksel_btHD.shape[1]
    assert vsel_btHD.shape == (B, Tk, H, D)
    assert kself_btHD.shape == (B, Tq, H, D)
    assert vself_btHD.shape == (B, Tq, H, D)
    assert sel_positions.shape[0] == Tk, "sel_positions must be length Tk"

    scale = (1.0 / math.sqrt(D)) if softmax_scale is None else float(softmax_scale)

    # contiguity for simple stride math
    q = q_btHD.contiguous()
    ks = ksel_btHD.contiguous()
    vs = vsel_btHD.contiguous()
    kS = kself_btHD.contiguous()
    vS = vself_btHD.contiguous()
    sel = sel_positions.contiguous().to(dtype=torch.int32, device=q.device)

    # outputs
    Tq_round = math.ceil(Tq / 128) * 128
    out = torch.empty_like(q)
    lse = torch.empty((B, H, Tq_round), device=q.device, dtype=torch.float32)

    # launch params
    BLOCK_D = max(16, triton.next_power_of_2(D))
    BLOCK_M = 128
    BLOCK_N = 128
    grid = (triton.cdiv(Tq, BLOCK_M), B * H)

    _fwd_kernel_scout_exact[grid](
        q, ks, vs, kS, vS, sel,
        out, lse,
        scale,
        # strides for (B,T,H,D)
        q.stride(0), q.stride(2), q.stride(1),
        ks.stride(0), ks.stride(2), ks.stride(1),
        vs.stride(0), vs.stride(2), vs.stride(1),
        kS.stride(0), kS.stride(2), kS.stride(1),
        vS.stride(0), vS.stride(2), vS.stride(1),
        sel.stride(0),
        out.stride(0), out.stride(2), out.stride(1),
        H, Tq, Tk, Tq_round, D,
        IS_CAUSAL=causal, DROP_DIAG=drop_diagonal,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=(4 if D <= 64 else 8),
        num_stages=1,
    )
    return out
