#!/usr/bin/env python3
"""Cross-window mixing at the trunk's own resolution, with optional MoE.

ContextIMUNet pools each window's trunk output -- 192 channels x 25 steps, 4800
numbers -- to a single 256-d vector before the mixer sees any of it, so the
mixer only ever reads a per-second summary. Round 6 is what points here:
widening the trunk 1.5x bought +0.0003 and halving it cost only +0.0049, and
widening the mixer, lengthening the context past 20 windows and training past
120 epochs were all null. Capacity is not the constraint; the pipe between the
two stages is.

Four knobs, all orthogonal, all off by default:

  fine        sub-steps each window contributes to the mixer. fine=1 reduces to
              ContextIMUNet's information flow, so this model contains it.
  moe         where the mixture of experts goes: "head" and "mlp" gate on the
              mixer output, "film" conditions the trunk from whole-segment
              statistics (the mixer runs after the trunk, so it cannot gate it).
  decompose   split the prediction into a segment-level term and a per-window
              deviation, mirroring what AVE and ATE20 separately measure.
  n_experts   how many experts, not necessarily four; the gate need not
              discover platforms.

`aux` is filled in on every forward with the gate logits and entropy. Entropy is
not optional bookkeeping: a gate collapsing onto one expert looks exactly like
"the mixture does not help", and without it the two cannot be told apart.
"""
from __future__ import annotations

import torch
EKF_RUNTIME = {"mask_direct": False}   # review 2026-09-20 night: on masked windows the fused output is the direct head (no learned carry); set from --ekf_mask_direct / checkpoint args
import torch.nn as nn
import torch.nn.functional as F

from unified.model import ResidualBlock, make_norm
from unified.moe import (ExpertHeads, ExpertMLP, FiLMGate, apply_film, gate_entropy,
                         segment_stats)


class FineContextIMUNet(nn.Module):
    """(B, K, 6, 200) -> velocity (B, K, 3), platform logits (B, K, 4)."""

    def __init__(self, fine: int = 5, hidden: int = 128, width: float = 1.0,
                 dropout: float = 0.2, d_model: int = 256, moe: str = "none",
                 n_experts: int = 4, decompose: bool = False, in_channels: int = 6,
                 continuous: bool = False, traj_film: bool = False, freq: bool = False,
                 norm: str = "bn", preint: bool = False, wide: bool = False,
                 wide_to_gru: bool = False, wide_span: int = 5, at: str = "none",
                 lr0: str = "none", head: str = "linear", drag: bool = False, vb_bins: int = 512, vb_range: float = 20.0, fuse: bool = False, ekf: bool = False, ekf_feats: bool = False):
        """continuous=True runs the trunk over the whole K-window segment as one
        200*K-frame signal instead of K separate seconds, so its 683-frame
        receptive field can use the neighbouring seconds; the per-second tokens
        are then cut out of the trunk's stride-8 output (25 steps per window).
        Same layers, same parameters, same GRU."""
        super().__init__()
        self.continuous = continuous
        # traj_film: FiLM the trunk on descriptors of the WHOLE trajectory (not
        # the 20 s segment): a legal form of test-time adaptation, since test
        # trajectories are given whole.  The descriptors are standardised with
        # buffers set from the training set so they travel with the weights.
        self.traj_film = traj_film
        from unified.moe import TRAJ_DESC_DIM
        n_ts = TRAJ_DESC_DIM if traj_film else 18            # 18 keeps every pre-2026-09-20 checkpoint loadable
        self.register_buffer("tstat_mean", torch.zeros(n_ts))
        self.register_buffer("tstat_std", torch.ones(n_ts))
        self.freq = freq
        # preint: the last 9 input channels are gyro pre-integration features
        # (updir.preint_features); they bypass the trunk through a small CNN and
        # enter the window summary through a zero-initialised projection
        self.preint = preint
        # wide: a second look at each second through its 5-second neighbourhood
        # (2 s either side, taken from the same segment, zero where the segment
        # or the valid length ends), low-passed at 20 Hz and decimated to 50 Hz,
        # through a small CNN, added to the summary via a zero-initialised map
        self.wide = wide
        # wide_span 1 (R3-1s-control): the same branch and filter, but the four
        # neighbouring windows are zeroed, so only the window's own second is seen
        self.wide_span = wide_span
        if not 1 <= fine <= 25:
            # adaptive_avg_pool1d handles any output length, including ones that
            # do not divide 25; an earlier divisibility check was my own and
            # silently ruled out every value between 1 and 5
            raise ValueError(f"fine must be between 1 and the trunk's 25 steps, got {fine}")
        if moe not in ("none", "head", "mlp", "film"):
            raise ValueError(f"moe must be none/head/mlp/film, got {moe!r}")
        self.fine, self.moe, self.decompose = fine, moe, decompose
        self.n_experts = n_experts
        self.aux: dict = {}

        c1, c2, c3, c4 = (max(8, int(round(c * width))) for c in (64, 96, 128, 192))
        # split into stages so FiLM has somewhere to attach; concatenated this is
        # exactly ContextIMUNet's encoder
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, c1, 7, padding=3, bias=False), make_norm(norm, c1), nn.GELU(),
            ResidualBlock(c1, c1, norm=norm))
        self.stage2 = nn.Sequential(ResidualBlock(c1, c2, stride=2, norm=norm),
                                    ResidualBlock(c2, c2, dilation=2, norm=norm))
        self.stage3 = nn.Sequential(ResidualBlock(c2, c3, stride=2, dilation=2, norm=norm),
                                    ResidualBlock(c3, c3, dilation=4, norm=norm))
        self.stage4 = nn.Sequential(ResidualBlock(c3, c4, stride=2, dilation=4, norm=norm),
                                    ResidualBlock(c4, c4, dilation=4, norm=norm))
        self.film = FiLMGate(18, n_experts, (c2, c3, c4)) if moe == "film" else None
        from unified.moe import TRAJ_DESC_DIM
        self.tfilm = FiLMGate(TRAJ_DESC_DIM, n_experts, (c2, c3, c4)) if traj_film else None
        # freq: log-magnitude spectrum of each window's channels, projected and
        # added to the window summary -- the frequency information the dilated
        # trunk has to reconstruct from time-domain kernels, handed over directly
        if wide:
            import numpy as _np
            taps = 31; n = _np.arange(taps) - taps // 2
            h = _np.sinc(2 * 20.0 / 200.0 * n) * _np.hamming(taps); h = h / h.sum()   # 20 Hz low-pass
            self.register_buffer("lp_kernel", torch.tensor(h, dtype=torch.float32).view(1, 1, -1).repeat(6, 1, 1))
            self.wide_net = nn.Sequential(
                nn.Conv1d(6, 32, 9, padding=4), nn.GELU(), nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.GELU(),
                nn.Conv1d(64, 64, 5, stride=2, padding=2), nn.GELU(), nn.AdaptiveAvgPool1d(1), nn.Flatten())
            self.wide_proj = nn.Linear(64, d_model)
            nn.init.zeros_(self.wide_proj.weight); nn.init.zeros_(self.wide_proj.bias)
        # wide_to_gru (C1, 2026-09-15): the same 5-second feature also enters the
        # five fine tokens of its second through a second zero-initialised map,
        # so the GRU can carry it across windows (wide alone only reaches the
        # summary, which is added after the GRU)
        # AT (2026-09-16, docs/schedules/training_schedule_adaptive_attention_2026-09-16.md):
        # one local residual cross-attention after the GRU. Query i = the GRU token
        # m_i (after mixer_proj); keys/values = the pre-GRU fine tokens e_j of the
        # windows i-2..i+2 (<= 25 keys, same segment, valid windows only). 4 heads x
        # 16 = 64 inner dims, LN on both sides, a learned bias over the 29 relative
        # fine-token offsets (-14..14) per head, W_O (64 -> d_model) zero-initialised
        # so the start reproduces the base model. "static": bias only (no content
        # Q/K); "content": Q/K on top of the same bias.
        self.at = at
        # the extra modules are built with the CPU RNG state forked and restored, so
        # every later shared tensor (shared, fine_proj, phase, GRU, heads) draws the
        # same initial values as a model without them (review 2026-09-16 13:03)
        rng_state = torch.get_rng_state()
        if at != "none":
            self.at_heads, self.at_dh = 4, 16
            inner = self.at_heads * self.at_dh
            self.at_ln_kv = nn.LayerNorm(d_model)
            self.at_v = nn.Linear(d_model, inner)
            self.at_o = nn.Linear(inner, d_model)
            nn.init.zeros_(self.at_o.weight); nn.init.zeros_(self.at_o.bias)
            self.at_bias = nn.Parameter(torch.zeros(self.at_heads, 4 * fine + 2 * (fine - 1) + 1))  # offsets -(2F+F-1)..+(2F+F-1)
            if at == "content":
                self.at_ln_q = nn.LayerNorm(d_model)
                self.at_q = nn.Linear(d_model, inner); self.at_k = nn.Linear(d_model, inner)
        torch.set_rng_state(rng_state)
        self.wide_tok = None
        if wide and wide_to_gru:
            self.wide_tok = nn.Linear(64, d_model)
            nn.init.zeros_(self.wide_tok.weight); nn.init.zeros_(self.wide_tok.bias)
        if preint:
            self.preint_net = nn.Sequential(
                nn.Conv1d(9, 32, 7, padding=3), nn.GELU(), nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.GELU(),
                nn.Conv1d(64, 64, 5, stride=2, padding=2), nn.GELU(), nn.AdaptiveAvgPool1d(1), nn.Flatten())
            self.preint_proj = nn.Linear(64, d_model)
            nn.init.zeros_(self.preint_proj.weight); nn.init.zeros_(self.preint_proj.bias)
        n_bins = 200 // 2 + 1
        self.freq_proj = nn.Sequential(
            nn.LayerNorm(in_channels * n_bins), nn.Linear(in_channels * n_bins, 256),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(256, d_model)) if freq else None

        self.attention = nn.Conv1d(c4, 1, 1)
        self.shared = nn.Sequential(nn.LayerNorm(3 * c4), nn.Linear(3 * c4, d_model),
                                    nn.GELU(), nn.Dropout(dropout))
        self.fine_proj = nn.Sequential(nn.LayerNorm(c4), nn.Linear(c4, d_model), nn.GELU())
        self.phase = nn.Parameter(torch.zeros(fine, d_model))
        nn.init.trunc_normal_(self.phase, std=0.02)

        self.mixer = nn.GRU(d_model, hidden, num_layers=2, batch_first=True, bidirectional=True)
        self.mixer_proj = nn.Linear(2 * hidden, d_model)
        self.fine_pool = nn.Linear(d_model, 1)
        self.mixer_norm = nn.LayerNorm(d_model)

        if moe == "head":
            self.velocity_head = ExpertHeads(d_model, n_experts, 3)
        elif moe == "mlp":
            self.velocity_head = ExpertMLP(d_model, n_experts, 3, dropout)
        else:
            self.velocity_head = nn.Linear(d_model, 3)
        # a segment-level term can only see the segment as a whole, which is what
        # makes the split structural rather than a reparametrisation
        self.segment_head = nn.Linear(d_model, 3) if decompose else None
        # VB-H-coordinate-v1 (2026-09-16, docs/reviews/distributional_head_feasibility_2026-09-16.md):
        # per-axis bin-expectation readout. Logits[axis, bin] = q_axis . e_bin with
        # q = W_q h (3 x 32 queries from the 256-d feature) and e_bin = W_c sin(gamma *
        # PE(b)), PE = 32 sine/cosine pairs of the physical centre b (frequencies
        # 10000^(-j/32)), gamma init ones. Softmax / expectation in float32; the
        # velocity is the expectation, trained with the existing vector Huber. Built
        # under a forked CPU RNG state so the common tensors match the linear-head model.
        self.head = head
        rng_state = torch.get_rng_state()
        if head == "vb":
            self.vb_bins, self.vb_range = vb_bins, float(vb_range)
            centres = torch.linspace(-vb_range, vb_range, vb_bins)                 # (N,)
            freq = 10000.0 ** (-torch.arange(32, dtype=torch.float32) / 32)      # (32,)
            pe = torch.cat([torch.sin(centres[:, None] * freq[None]), torch.cos(centres[:, None] * freq[None])], 1)  # (N, 64)
            self.register_buffer("vb_centres", centres); self.register_buffer("vb_pe", pe)
            self.vb_gamma = nn.Parameter(torch.ones(64))
            self.vb_coord = nn.Linear(64, 32)
            self.vb_query = nn.Linear(d_model, 3 * 32)
        torch.set_rng_state(rng_state)
        # LR0-v1 (2026-09-16, docs/reviews/relative_motion_next_design_2026-09-16.md):
        # a zero-initialised, bias-free per-window linear map e = W_e h added to the
        # velocity. "direct": q = p + e. "graph": within each fixed 5-window block of
        # the chunk, q = p + S_n e with S_n = (I + D_n^T D_n)^-1 D_n^T D_n (D_n the
        # first-difference operator over the n valid windows of the block, lambda = 1):
        # the correction changes the within-block differences and preserves the block
        # mean of the current p; n = 1 gives no correction.
        # road 3 (2026-09-18): drag-structured output. A recording-level linear map from
        # the window-mean raw IMU (specific force, gyro; standardised units) to velocity,
        # coefficients estimated from the chunk-pooled features, gated per window.
        # v = head(h_k) + sigmoid(gate(h_k)) * (A u_k + b), u_k = [f_mean, w_mean]_k.
        # A, b and the gate are zero-initialised (gate 0.5) so the start equals the base.
        self.drag = drag
        if drag:
            self.drag_coef = nn.Linear(d_model, 3 * 6 + 3); nn.init.zeros_(self.drag_coef.weight); nn.init.zeros_(self.drag_coef.bias)
            self.drag_gate = nn.Linear(d_model, 3); nn.init.zeros_(self.drag_gate.weight); nn.init.zeros_(self.drag_gate.bias)
        # learned-INS step A (2026-09-19): gated fusion of the dead-reckoned velocity channel.
        # v = v_direct + sigmoid(gate(h_k)) * (v_DR_k - v_direct); gate bias -3 (sigma ~ 0.05)
        # and zero weight so the start equals the direct head; v_DR_k is the un-normalised
        # window mean passed in by the caller (m/s).
        # beyond-deadline road 1 stage 1 (2026-09-20): the gate also reads three per-window
        # scalars from the up stream (t_since_anchor, rest-window score, |v_DR|) so it can learn
        # WHEN v_DR is trustworthy; supervised by the trainer (--fuse_sup) with 1[|v_DR - y| < thr].
        self.fuse = fuse
        if fuse:
            self.fuse_gate = nn.Linear(d_model + 3, 3); nn.init.zeros_(self.fuse_gate.weight); nn.init.constant_(self.fuse_gate.bias, -3.0)
        # road 1 stage 2 (2026-09-20): learned EKF over the chunk. v_k = v_prop,k + K_k * (v_direct,k -
        # v_prop,k), v_prop,k = dR_k v_{k-1} + dV_k (IMU-only physics increments from the derived
        # channels), K_k = sigmoid(gain(h_k, gfeat_k)); v_0 = v_direct,0. Gain bias +3 (sigma 0.95):
        # the start is the direct head; --ekf_sup teaches K from which of the two is closer to y.
        self.ekf = ekf; self.ekf_feats = ekf_feats
        if ekf:
            self.ekf_gain = nn.Linear(d_model + 3, 3); nn.init.zeros_(self.ekf_gain.weight); nn.init.constant_(self.ekf_gain.bias, 3.0)
        if ekf and ekf_feats:
            # advice_after_v20 §2.2/2.3: the gain also reads the increment confidence mask, the
            # per-recording calibration deviation (/10) and, per step, the innovation
            # log1p|v_direct,k - v_prop,k| (own linear over [mask, innovation, conf], zero-init: starts as a3v20)
            self.ekf_gain2 = nn.Linear(3, 3); nn.init.zeros_(self.ekf_gain2.weight); nn.init.zeros_(self.ekf_gain2.bias)
        self.lr0 = lr0
        self.anchor_mu = 5.0                      # --lr0 anchor: increment-vs-anchor weight (set by the trainer)
        rng_state = torch.get_rng_state()
        if lr0 != "none":
            self.lr0_head = nn.Linear(d_model, 3, bias=False)
            nn.init.zeros_(self.lr0_head.weight)
            for n in range(1, 6):
                D = torch.zeros(max(n - 1, 0), n)
                for i in range(n - 1):
                    D[i, i] = -1.0; D[i, i + 1] = 1.0
                S = torch.linalg.solve(torch.eye(n) + D.T @ D, D.T @ D) if n > 1 else torch.zeros(1, 1)
                self.register_buffer(f"lr0_S{n}", S)
            table = torch.zeros(6, 5, 5)
            for n in range(1, 6):
                table[n, :n, :n] = getattr(self, f"lr0_S{n}")
            self.register_buffer("lr0_table", table)
        torch.set_rng_state(rng_state)
        self.platform_head = nn.Linear(d_model, 4)

    def trunk_seq(self, x, gammas=None, betas=None, repeat=1):
        h = self.stem(x)
        for i, stage in enumerate((self.stage2, self.stage3, self.stage4)):
            h = stage(h)
            if gammas is not None:
                h = apply_film(h, gammas[i], betas[i], repeat)
        return h

    def _summary(self, h):
        w = torch.softmax(self.attention(h), dim=-1)
        return torch.cat([(h * w).sum(-1), h.mean(-1), h.std(-1, unbiased=False)], dim=1)

    def _trunk_continuous(self, x, lengths, valid):
        """(B, K, C, T) -> per-window trunk output (n, C4, T//8), n = valid windows.

        Segments are run grouped by valid length so the trunk never sees a padded
        window: each group is (b, C, L*T) of real, contiguous signal.  The
        output rows are ordered like x[valid] (batch-major, then window)."""
        B, K, C, T = x.shape
        if valid is None:
            h = self.trunk_seq(x.transpose(1, 2).reshape(B, C, K * T))
            return h.reshape(B, h.shape[1], K, -1).transpose(1, 2).reshape(B * K, h.shape[1], -1)
        out = [None] * B
        for L in torch.unique(lengths).tolist():
            rows = (lengths == L).nonzero(as_tuple=True)[0]
            xs = x[rows, :L].transpose(1, 2).reshape(len(rows), C, L * T)
            h = self.trunk_seq(xs)                                  # (b, C4, L*25)
            h = h.reshape(len(rows), h.shape[1], L, -1).transpose(1, 2)   # (b, L, C4, 25)
            for r, hr in zip(rows.tolist(), h):
                out[r] = hr
        return torch.cat(out, dim=0)

    def _neighbourhood(self, x, valid):
        """(B, K, C, T) -> (n_valid or B*K, 6, 5T/4): each window's raw six axes with
        two windows either side, zeroed outside the segment / valid length,
        low-passed and decimated by 4."""
        B, K, C, T = x.shape
        raw = x[:, :, :6]                                              # (B, K, 6, T)
        if valid is not None:
            raw = raw * valid[:, :, None, None].to(raw.dtype)
        pad = raw.new_zeros(B, 2, 6, T)
        ext = torch.cat([pad, raw, pad], dim=1)                        # (B, K+4, 6, T)
        neigh = torch.stack([ext[:, i:i + K] for i in range(5)], dim=3)  # (B, K, 6, 5, T)
        if self.wide_span == 1:
            keep = neigh.new_zeros(5); keep[2] = 1
            neigh = neigh * keep[None, None, None, :, None]
        neigh = neigh.reshape(B, K, 6, 5 * T)
        neigh = neigh.reshape(B * K, 6, 5 * T) if valid is None else neigh[valid]
        lp = F.conv1d(neigh.float(), self.lp_kernel, padding=self.lp_kernel.shape[-1] // 2, groups=6)
        return lp[:, :, ::4].to(x.dtype)                                # 50 Hz

    def _local_attention(self, q_tok, kv_tok, valid):
        """q_tok (B, K, F, d) GRU tokens m_i; kv_tok (B, K, F, d) pre-GRU tokens e_j;
        valid (B, K) bool or None. Returns the residual W_O(sum_j a_ij W_V LN(e_j))."""
        B, K, Fn, d = q_tok.shape
        H, dh = self.at_heads, self.at_dh
        kv = self.at_ln_kv(kv_tok)
        pad = kv.new_zeros(B, 2, Fn, d)
        ext = torch.cat([pad, kv, pad], dim=1)                                       # (B, K+4, F, d)
        keys = torch.stack([ext[:, r:r + K] for r in range(5)], dim=2).reshape(B, K, 5 * Fn, d)
        vmask = torch.ones(B, K, dtype=torch.bool, device=q_tok.device) if valid is None else valid
        vext = torch.cat([vmask.new_zeros(B, 2), vmask, vmask.new_zeros(B, 2)], dim=1)
        kmask = torch.stack([vext[:, r:r + K] for r in range(5)], dim=2)            # (B, K, 5)
        kmask = kmask[:, :, :, None].expand(B, K, 5, Fn).reshape(B, K, 1, 1, 5 * Fn)  # (B,K,1,1,5F)
        # relative fine-token offset of key (r, g) from query f: (r-2)*F + g - f  in [-(3F-1), 3F-1]
        r = torch.arange(5, device=q_tok.device); g = torch.arange(Fn, device=q_tok.device); f = torch.arange(Fn, device=q_tok.device)
        off = ((r[:, None] - 2) * Fn + g[None, :]).reshape(-1)[None, :] - f[:, None]  # (F, 5F)
        bias = self.at_bias[:, off + (3 * Fn - 1)]                                  # (H, F, 5F)
        logits = bias[None, None]                                                    # (1,1,H,F,5F)
        v = self.at_v(keys).reshape(B, K, 5 * Fn, H, dh)
        if self.at == "content":
            q = self.at_ln_q(q_tok)
            qh = self.at_q(q).reshape(B, K, Fn, H, dh)
            kh = self.at_k(keys).reshape(B, K, 5 * Fn, H, dh)
            logits = logits + torch.einsum("bkfhd,bkghd->bkhfg", qh.float(), kh.float()).to(q_tok.dtype) / (dh ** 0.5)
        else:
            logits = logits.expand(B, K, H, Fn, 5 * Fn)
        # masked keys get a large finite negative logit (no -inf/NaN in backward);
        # a query whose keys are all masked (padded window) has its output zeroed
        logits = logits.masked_fill(~kmask, -1e4)
        w = torch.softmax(logits.float(), dim=-1).to(q_tok.dtype)
        out = torch.einsum("bkhfg,bkghd->bkfhd", w, v).reshape(B, K, Fn, H * dh)
        out = out * vmask[:, :, None, None].to(out.dtype)
        return self.at_o(out)

    def _spectrum(self, xw):
        """(n, C, T) -> (n, C*(T//2+1)) log-magnitude rfft, DC included."""
        mag = torch.fft.rfft(xw.float(), dim=-1).abs()
        return torch.log1p(mag).flatten(1).to(xw.dtype)

    def features_and_heads(self, x, lengths=None, tstats=None, window_mask=None, mask_tokens=None, vdr=None, gfeat=None, dv=None, dR=None, emask=None):
        """lengths (B,) marks how many of the K windows are real. Padded windows
        never reach the trunk (so BatchNorm statistics are clean) and the GRU is
        packed to the valid length, so a padded position cannot leak backwards
        into a real one through the reverse direction. Outputs at padded
        positions are undefined; the caller masks them.

        window_mask (B, K) bool with mask_tokens = (tok_f (fine, d), tok_s (d)):
        segment-level pre-training. A masked window never enters the trunk; its
        fine tokens and its summary are replaced by the learned tokens, so the
        only way to predict it is from the neighbouring windows through the GRU."""
        B, K = x.shape[:2]
        self.aux = {}
        x_pre = None
        if self.preint:
            x_pre, x = x[:, :, -9:], x[:, :, :-9]
        valid = None
        if lengths is not None and bool((lengths < K).any()):
            valid = torch.arange(K, device=x.device)[None, :] < lengths[:, None]   # (B, K)
        if window_mask is not None:
            if self.continuous or self.film is not None or self.tfilm is not None:
                raise NotImplementedError("window_mask with continuous/FiLM")
            valid = (~window_mask) if valid is None else (valid & ~window_mask)

        gammas = betas = None
        if self.film is not None:
            gammas, betas, logits, w = self.film(segment_stats(x))
            self.aux["gate_logits"] = logits            # (B, K_experts), segment-level
            self.aux["gate_entropy"] = gate_entropy(w)
        if self.tfilm is not None:
            if tstats is None:
                raise ValueError("traj_film model needs tstats")
            z = (tstats - self.tstat_mean) / self.tstat_std
            gammas, betas, logits, w = self.tfilm(z)
            self.aux["gate_logits"] = logits
            self.aux["gate_entropy"] = gate_entropy(w)

        if self.continuous:
            if gammas is not None:
                raise NotImplementedError("FiLM with a continuous trunk")
            h = self._trunk_continuous(x, lengths, valid)   # (n_valid or B*K, C, 25)
            xw = x.reshape(B * K, *x.shape[2:]) if valid is None else x[valid]
        elif valid is None:
            xw = x.reshape(B * K, *x.shape[2:])
            h = self.trunk_seq(xw, gammas, betas, repeat=K)
        else:
            xw = x[valid]                               # (n_valid, C, T)
            rows = torch.arange(B, device=x.device)[:, None].expand(B, K)[valid]
            h = self.trunk_seq(xw, gammas, betas, repeat=rows)
        summary = self.shared(self._summary(h))
        if self.freq_proj is not None:
            summary = summary + self.freq_proj(self._spectrum(xw))
        if x_pre is not None:
            xp = x_pre.reshape(B * K, 9, -1) if valid is None else x_pre[valid]
            summary = summary + self.preint_proj(self.preint_net(xp))
        if self.wide:
            bw = self.wide_net(self._neighbourhood(x, valid))
            summary = summary + self.wide_proj(bw)

        f = F.adaptive_avg_pool1d(h, self.fine)
        f = self.fine_proj(f.transpose(1, 2)) + self.phase
        if self.wide_tok is not None:
            f = f + self.wide_tok(bw)[:, None, :]
        if valid is not None:                           # scatter back to (B, K, ...)
            full = f.new_zeros(B, K, *f.shape[1:]); full[valid] = f; f = full
            full = summary.new_zeros(B, K, summary.shape[-1]); full[valid] = summary
            summary = full
            if window_mask is not None:
                tok_f, tok_s = mask_tokens
                f = torch.where(window_mask[:, :, None, None], tok_f.to(f.dtype)[None, None], f)
                summary = torch.where(window_mask[:, :, None], tok_s.to(summary.dtype)[None, None], summary)
        seq = f.reshape(B, K * self.fine, -1)
        if lengths is None or not bool((lengths < K).any()):
            mixed, _ = self.mixer(seq)
        else:
            packed = nn.utils.rnn.pack_padded_sequence(
                seq, (lengths * self.fine).cpu(), batch_first=True, enforce_sorted=False)
            mixed, _ = self.mixer(packed)
            mixed, _ = nn.utils.rnn.pad_packed_sequence(mixed, batch_first=True,
                                                        total_length=K * self.fine)
        mixed = self.mixer_proj(mixed).reshape(B, K, self.fine, -1)
        if self.at != "none":
            mixed = mixed + self._local_attention(mixed, f.reshape(B, K, self.fine, -1), valid)
        w_fine = torch.softmax(self.fine_pool(mixed), dim=2)
        features = self.mixer_norm(summary.view(B, K, -1) + (mixed * w_fine).sum(2))

        if self.moe in ("head", "mlp"):
            velocity, logits, w = self.velocity_head(features)
            self.aux["gate_logits"] = logits            # (B, K, K_experts), per window
            self.aux["gate_entropy"] = gate_entropy(w)
        elif self.head == "vb":
            velocity = self._vb_velocity(features)
        else:
            velocity = self.velocity_head(features)

        if self.drag:
            u = x[:, :, :6, :].float().mean(-1)                                     # (B, K, 6) window-mean raw IMU
            if valid is None:
                h_rec = features.float().mean(1)
            else:
                h_rec = (features.float() * valid[..., None]).sum(1) / lengths[:, None].clamp(min=1)
            coef = self.drag_coef(h_rec)                                            # (B, 21)
            A, bias = coef[:, :18].reshape(B, 3, 6), coef[:, 18:]                    # (B,3,6), (B,3)
            lin = torch.einsum('bij,bkj->bki', A, u) + bias[:, None, :]              # (B, K, 3)
            gate = torch.sigmoid(self.drag_gate(features).float())
            self.aux["drag_A"] = A; self.aux["drag_lin"] = lin; self.aux["drag_gate"] = gate
            velocity = velocity + (gate * lin).to(velocity.dtype)
        if self.fuse and vdr is not None:
            if gfeat is None:
                gfeat = torch.zeros(features.shape[0], features.shape[1], 3, dtype=features.dtype, device=features.device)
            logit = self.fuse_gate(torch.cat([features, gfeat.to(features.dtype)], dim=-1)).float()   # (B, K, 3)
            g = torch.sigmoid(logit)
            self.aux["fuse_vdirect"] = velocity; self.aux["fuse_gate"] = g; self.aux["fuse_logit"] = logit
            velocity = velocity + (g * (vdr.float() - velocity.float())).to(velocity.dtype)
        if self.ekf and dv is not None and dR is not None:
            if gfeat is None:
                gfeat = torch.zeros(features.shape[0], features.shape[1], 3, dtype=features.dtype, device=features.device)
            logit = self.ekf_gain(torch.cat([features, gfeat.to(features.dtype)], dim=-1)).float()   # (B, K, 3)
            vd = velocity.float(); dv = dv.float(); dR = dR.float()
            if self.ekf_feats:
                em = (emask if emask is not None else torch.cat([torch.ones_like(vd[..., :1]), torch.zeros_like(vd[..., :1])], -1)).float()   # (B, K, 2) = [mask, conf/10]
                w2 = self.ekf_gain2.weight.float(); b2 = self.ekf_gain2.bias.float()          # (3, 3), (3,)
                base = logit + em[..., :1] * w2[:, 0] + em[..., 1:2] * w2[:, 2] + b2
                vs = [vd[:, 0]]; props = [vd[:, 0]]; lgs = [base[:, 0]]
                md = EKF_RUNTIME["mask_direct"]
                for k in range(1, vd.shape[1]):
                    vprop = (dR[:, k] @ vs[-1][..., None])[..., 0] + dv[:, k]
                    innov = torch.log1p((vd[:, k] - vprop).norm(dim=-1, keepdim=True))       # (B, 1)
                    lg = base[:, k] + innov * w2[:, 1]
                    if md:
                        lg = lg + 6.0 * (1.0 - em[:, k, :1])                                    # masked window -> gain ~1 -> direct head
                    lgs.append(lg); props.append(vprop)
                    vs.append(vprop + torch.sigmoid(lg) * (vd[:, k] - vprop))
                logit = torch.stack(lgs, 1)
            else:
                if EKF_RUNTIME["mask_direct"] and emask is not None:
                    logit = logit + 6.0 * (1.0 - emask[..., :1].float())
                Kg0 = torch.sigmoid(logit)
                vs = [vd[:, 0]]; props = [vd[:, 0]]
                for k in range(1, vd.shape[1]):
                    vprop = (dR[:, k] @ vs[-1][..., None])[..., 0] + dv[:, k]
                    props.append(vprop)
                    vs.append(vprop + Kg0[:, k] * (vd[:, k] - vprop))
            Kg = torch.sigmoid(logit)
            vf = torch.stack(vs, 1); vp = torch.stack(props, 1)
            self.aux["ekf_vdirect"] = velocity; self.aux["ekf_gain"] = Kg; self.aux["ekf_logit"] = logit; self.aux["ekf_vprop"] = vp
            velocity = vf.to(velocity.dtype)
        if self.segment_head is not None:
            if valid is None:
                seg = features.mean(dim=1, keepdim=True)
            else:
                seg = (features * valid[..., None]).sum(1, keepdim=True) / lengths[:, None, None]
            velocity = velocity + self.segment_head(seg)
        if self.lr0 == "anchor":
            velocity = self._anchored_increments(features, velocity, valid, self.anchor_mu)
        elif self.lr0 != "none":
            velocity = velocity + self._lr0_correction(features, valid)
        return features, velocity, self.platform_head(features)

    def _anchored_increments(self, features, p, valid, mu: float = 1.0):
        """Method 2 (2026-09-17): the direct head p_k is the anchor, the increment
        head predicts e_k ~ y_k - y_{k-1}; the chunk velocity is the closed-form
        q = argmin sum_k w_k |q_k - p_k|^2 + mu sum_k |(q_k - q_{k-1}) - e_k|^2 with
        anchor weights w_k = 1/(1+|p_k|) (slow windows anchor, fast windows are
        reached by accumulating in-distribution increments).  Solved per chunk in
        float32; invalid (padded) windows keep q_k = p_k.  p and e are kept on the
        module for the trainer's anchor / increment losses."""
        B, K, _ = features.shape
        e = self.lr0_head(features)                                       # (B, K, 3)
        vmask = torch.ones(B, K, dtype=torch.bool, device=features.device) if valid is None else valid
        with torch.autocast(device_type=features.device.type, enabled=False):
            p32, e32, m = p.float(), e.float(), vmask.float()
            e32 = e32 * m[..., None]
            w = 1.0 / (1.0 + p32.detach().norm(dim=-1))                   # (B, K) anchor weights
            w = torch.where(vmask, w, torch.ones_like(w))
            # D: (K-1, K) first differences; rows touching an invalid window are dropped
            D = torch.zeros(K - 1, K, device=p.device); idx = torch.arange(K - 1, device=p.device)
            D[idx, idx] = -1.0; D[idx, idx + 1] = 1.0
            rowok = (m[:, 1:] * m[:, :-1])                                # (B, K-1)
            Db = D[None] * rowok[..., None]                               # (B, K-1, K)
            A = torch.diag_embed(w) + mu * Db.transpose(1, 2) @ Db        # (B, K, K)
            rhs = w[..., None] * p32 + mu * Db.transpose(1, 2) @ e32[:, 1:]   # (B, K, 3); e_k pairs (k-1, k)
            q = torch.linalg.solve(A, rhs)
        self._anchor = (p, e)
        return q.to(p.dtype)

    def _vb_velocity(self, features):
        """(B, K, d) -> (B, K, 3): expectation of a per-axis bin distribution (float32)."""
        with torch.autocast(device_type=features.device.type, enabled=False):
            f = features.float()
            q = self.vb_query(f).reshape(*f.shape[:-1], 3, 32)                       # (B, K, 3, 32)
            e = self.vb_coord(torch.sin(self.vb_gamma * self.vb_pe))                 # (N, 32)
            logits = torch.einsum("bkad,nd->bkan", q, e)                              # (B, K, 3, N)
            prob = torch.softmax(logits, dim=-1)
            return (prob * self.vb_centres).sum(-1)                                   # (B, K, 3)

    def _lr0_correction(self, features, valid):
        """(B, K, d) -> (B, K, 3): the LR0 correction (see __init__)."""
        B, K, _ = features.shape
        e = self.lr0_head(features)                                       # (B, K, 3)
        vmask = torch.ones(B, K, dtype=torch.bool, device=features.device) if valid is None else valid
        e = e * vmask[..., None].to(e.dtype)
        if self.lr0 == "direct":
            return e
        # the <= 5x5 projection is done in float32 with autocast off (review
        # 2026-09-16 12:34): no fp16 DC leakage into the preserved block mean.
        # Vectorised: block b of row i uses S_n with n = clamp(L_i - 5b, 0, 5)
        # (prefix-valid windows), looked up in a (6, 5, 5) table.
        with torch.autocast(device_type=e.device.type, enabled=False):
            e32 = e.float()
            nblk = (K + 4) // 5
            padK = nblk * 5
            if padK != K:
                e32 = torch.nn.functional.pad(e32, (0, 0, 0, padK - K))
            eb = e32.reshape(B, nblk, 5, 3)
            L = vmask.sum(1)                                              # (B,) prefix-valid windows
            n = (L[:, None] - 5 * torch.arange(nblk, device=e.device)[None, :]).clamp(0, 5)   # (B, nblk)
            table = self.lr0_table                                        # (6, 5, 5)
            Sb = table[n]                                                 # (B, nblk, 5, 5)
            out = torch.einsum("bkij,bkjc->bkic", Sb, eb).reshape(B, padK, 3)[:, :K]
        return out.to(e.dtype)

    def adapt_tstat(self, state_dict, prefix="net."):
        """resize the tstat buffers to a checkpoint's shape before load_state_dict (any size)."""
        k = prefix + "tstat_mean"
        if k in state_dict and state_dict[k].shape != self.tstat_mean.shape:
            n = state_dict[k].shape[0]; dev = self.tstat_mean.device
            self.tstat_mean = torch.zeros(n, device=dev); self.tstat_std = torch.ones(n, device=dev)

    def forward(self, x, lengths=None, tstats=None, vdr=None, gfeat=None, dv=None, dR=None, emask=None):
        _, velocity, platform_logits = self.features_and_heads(x, lengths, tstats, vdr=vdr, gfeat=gfeat, dv=dv, dR=dR, emask=emask)
        return velocity, platform_logits
