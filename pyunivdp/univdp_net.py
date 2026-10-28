"""The UniVDP network.

DINOv3 features of the test and the reference (keyframe at high resolution, a 250 ms clip at low resolution) are compared at three
depths of the backbone. At each depth, gated local pooling selects where the feature differences matter: the gate of the keyframe
also sees the ColorVideoVDP visibility map, the gate of the clip's temporal tokens sees the ColorVideoVDP transient channel. The
temporal tokens are modulated (FiLM) by the frame rates and by psychophysical motion terms (judder, hold blur, flicker, motion
masking) of the reference motion speed. Self-attention per depth, cross-depth attention and attention pooling (with ColorVideoVDP
band tokens) give one score per scored position.

The attention head adapts CFANet of TOPIQ (C. Chen et al., "TOPIQ: A Top-down Approach from Semantics to Distortions for Image
Quality Assessment", IEEE TIP 2024), from the IQA-PyTorch implementation (pyiqa/archs/topiq_arch.py, Copyright 2022 S-Lab,
S-Lab License 1.0, see THIRD_PARTY_NOTICES.md).

Parameter names match the training code, so its checkpoints (``model_state_dict``) load into ``UniVDPNet`` directly.
"""
import copy
import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------------------------------------------------------------
# CFANet building blocks (IQA-PyTorch, S-Lab License 1.0): pre-norm transformer layers, sequence-first.
# ---------------------------------------------------------------------------------------------------------------------------------
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, value=src2)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(F.gelu(self.linear1(src2))))
        return src + self.dropout2(src2)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)       # unused in the forward (as in CFANet)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory):
        memory = self.norm2(memory)
        tgt2 = self.norm1(tgt)
        tgt2, _ = self.multihead_attn(query=tgt2, key=memory, value=memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(F.gelu(self.linear1(tgt2))))
        return tgt + self.dropout3(tgt2)


class TransformerStack(nn.Module):
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])

    def forward(self, x, *memory):
        for layer in self.layers:
            x = layer(x, *memory)
        return x


class GatedPool(nn.Module):
    """Gate of the gated local pooling: per location in [0, 1], from global self-attention over the feature differences, plus a
    1x1 projection of a ColorVideoVDP map (visibility or transient) added to the logit."""

    def __init__(self, in_ch, c=64, nhead=4, hidden=64):
        super().__init__()
        self.norm = nn.LayerNorm(in_ch)
        self.attn = nn.MultiheadAttention(in_ch, nhead, batch_first=True)
        self.mlp = nn.Sequential(nn.LayerNorm(in_ch), nn.Linear(in_ch, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.cv_proj = nn.Conv2d(c, 1, 1)

    def forward(self, x, cvmap):                                    # x [B, C, H, W] feature differences, cvmap [B, c, h, w]
        B, C, H, W = x.shape
        t = x.flatten(2).transpose(1, 2)
        tn = self.norm(t)
        a, _ = self.attn(tn, tn, tn)
        logit = self.mlp(t + a)                                     # [B, HW, 1]
        cm = cvmap if cvmap.shape[-2:] == (H, W) else F.interpolate(cvmap, size=(H, W), mode='bilinear', align_corners=False)
        logit = logit + self.cv_proj(cm.to(x.dtype)).flatten(2).transpose(1, 2)
        return torch.sigmoid(logit).transpose(1, 2).reshape(B, 1, H, W)


class CFANet(nn.Module):
    """The feature extractor and attention head of CFANet with a DINOv3 backbone cut after the last tapped block."""

    def __init__(self, backbone='vit_small_patch16_dinov3', taps=(2, 4, 7), inter_dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.semantic_model = timm.create_model(backbone, pretrained=False, features_only=True, out_indices=tuple(taps))
        dims = self.semantic_model.feature_info.channels()
        ffn = min(4 * inter_dim, 2048)
        act = nn.GELU()
        self.weight_pool = nn.ModuleList([GatedPool(d) for d in dims])
        self.dim_reduce = nn.ModuleList([nn.Sequential(nn.Conv2d(3 * d, inter_dim, 1, 1), act) for d in dims])
        self.sa_attn_blks = nn.ModuleList([TransformerStack(TransformerEncoderLayer(inter_dim, num_heads, ffn, dropout), 1) for _ in dims])
        self.attn_blks = nn.ModuleList([TransformerStack(TransformerDecoderLayer(inter_dim, num_heads, ffn, dropout), 1)
                                        for _ in range(len(dims) - 1)])
        self.attn_pool = TransformerEncoderLayer(inter_dim, num_heads, ffn, dropout)
        self.score_linear = nn.Sequential(nn.LayerNorm(inter_dim), nn.Linear(inter_dim, inter_dim), act,
                                          nn.LayerNorm(inter_dim), nn.Linear(inter_dim, inter_dim), act, nn.Linear(inter_dim, 1))
        self.h_emb = nn.Parameter(torch.zeros(1, inter_dim // 2, 32, 1))
        self.w_emb = nn.Parameter(torch.zeros(1, inter_dim // 2, 1, 32))
        self.register_buffer('mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def features(self, x):                                          # x [N, 3, H, W] in [0, 1]
        return self.semantic_model((x - self.mean.to(x)) / self.std.to(x))

    @staticmethod
    def dist_func(x, y, eps=1e-12):
        return torch.sqrt((x - y) ** 2 + eps)


# ---------------------------------------------------------------------------------------------------------------------------------
# ColorVideoVDP and psychophysical inputs
# ---------------------------------------------------------------------------------------------------------------------------------
def cvvdp_band_freq(ppd, k):
    """Centre frequency (cycles per degree) of ColorVideoVDP pyramid band k (band 0 = finest)."""
    return ppd / 2 * (1.0 if k == 0 else 0.3228 * 2.0 ** -(k - 1))


class VisibilityEncoder(nn.Module):
    """Per-window ColorVideoVDP statistics of every band ([B, Hb, Wb, n]) -> one map [B, c, H, W] on the DINO token grid: log1p,
    a shared linear layer plus an embedding of the band's centre frequency (log2 cpd), resampled to the grid (area pooling when
    finer, bilinear when coarser), then a per-location softmax over the bands."""

    def __init__(self, c=64, in_dim=18):
        super().__init__()
        self.proj = nn.Linear(in_dim, c)
        self.freq = nn.Sequential(nn.Linear(1, c), nn.GELU(), nn.Linear(c, c))
        self.score = nn.Linear(c, 1)
        self.out = nn.Linear(c, c)

    def forward(self, wins, ppd, H, W):
        fs, ss = [], []
        for k, w in enumerate(wins):
            e = self.freq(torch.log2(cvvdp_band_freq(ppd.float(), k)).unsqueeze(-1))                  # [B, c]
            f = F.gelu(self.proj(torch.log1p(w.float().clamp_min(0))) + e[:, None, None, :])          # [B, Hb, Wb, c]
            f = f.permute(0, 3, 1, 2)
            if f.shape[-2] >= H and f.shape[-1] >= W:
                f = F.adaptive_avg_pool2d(f, (H, W))
            else:
                f = F.interpolate(f, size=(H, W), mode='bilinear', align_corners=False)
            fs.append(f)
            ss.append(self.score(f.permute(0, 2, 3, 1)))                                             # [B, H, W, 1]
        a = torch.softmax(torch.stack(ss, 0), 0).permute(0, 1, 4, 2, 3)                              # [nb, B, 1, H, W]
        C = (torch.stack(fs, 0) * a).sum(0)
        return self.out(C.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class PsyMotion(nn.Module):
    """Psychophysical motion terms from the reference speed v (deg/s) and the test / reference frame rates, each a sigmoid in log
    units with a learned threshold and slope (initialised at literature magnitudes):
      judder   sigmoid(k (log2(v / F_test) - th)), th0 = log2(0.25 deg per test frame)
      blur     sigmoid(k (log2(v / F_test) - th)), th0 = log2(0.05 deg)     (hold-type blur extent ~ v / F)
      flicker  sigmoid(k (th - log2 F_test)),      th0 = log2(50 Hz)
      masking  sigmoid(k (log2 v - th)),           th0 = log2(10 deg/s)
    plus the frame-rate drop log2(F_ref / F_test)."""

    def __init__(self):
        super().__init__()
        self.th = nn.Parameter(torch.tensor([math.log2(0.25), math.log2(0.05), math.log2(50.0), math.log2(10.0)]))
        self.k = nn.Parameter(torch.full((4,), 0.5413))                                              # softplus(0.5413) = 1

    def forward(self, ft, fr, v):
        lv, lft = torch.log2(v.clamp_min(1e-3)), torch.log2(ft)
        ld = lv - lft
        k = F.softplus(self.k)
        x = torch.stack([ld - self.th[0], ld - self.th[1], self.th[2] - lft, lv - self.th[3]], -1)
        return torch.cat([torch.sigmoid(k * x), (torch.log2(fr) - lft).unsqueeze(-1)], -1)          # [B, 5]


# ---------------------------------------------------------------------------------------------------------------------------------
# The full model
# ---------------------------------------------------------------------------------------------------------------------------------
class UniVDPNet(nn.Module):
    """forward(test, ref, band, wins, ppd, clip=None) -> score [B] for one scored position (an image, or a video keyframe with its
    250 ms clip). Inputs:
      test, ref  [B, 3, H, W] display-encoded RGB in [0, 1], H and W multiples of 16
      band       [B, nb, 36] ColorVideoVDP band statistics of the keyframe (window mean and standard deviation of 18 statistics)
      wins       list over bands of [B, Hb, Wb, 18] ColorVideoVDP window statistics of the keyframe (visibility map)
      ppd        [B] pixels per degree of the native frame
      clip       None for images, else a dict:
                   t, r     [B, M, 3, h, w] clip frames in [0, 1] (short side 224)
                   pairs    [B, P, 2] indices of adjacent-frame pairs into the M frames
                   tcond    [B, 4] log2(F_test / 30), log2(F_ref / 30), log2(frame step / 33.3 ms), reference speed (deg/s)
                   tr_win   list over bands of [B, Hb, Wb, 12] ColorVideoVDP transient-channel window statistics
                   tr_ppd   [B] pixels per degree of the transient statistics"""

    def __init__(self, backbone='vit_small_patch16_dinov3', taps=(2, 4, 7), d=256, c=64):
        super().__init__()
        self.cfanet = CFANet(backbone, taps, inter_dim=d)
        dims = self.cfanet.semantic_model.feature_info.channels()
        self.cv_band_proj = nn.Linear(36, d)                       # visibility tokens (attention pooling)
        self.cv_band_pos = nn.Embedding(16, d)
        self.cv_type = nn.Parameter(torch.zeros(d))
        self.cv_enc = VisibilityEncoder(c, in_dim=18)               # visibility map -> keyframe gates
        self.tr_enc = VisibilityEncoder(c, in_dim=12)               # transient map -> temporal gates
        self.t_gate = nn.ModuleList([GatedPool(ch, c) for ch in dims])
        self.t_dim_reduce = nn.ModuleList([nn.Sequential(nn.Conv2d(3 * ch, d, 1), nn.GELU()) for ch in dims])
        self.t_type = nn.Parameter(torch.zeros(d))
        self.psy = PsyMotion()
        self.t_film = nn.Sequential(nn.Linear(8, 64), nn.GELU(), nn.Linear(64, 2 * d * len(dims)))

    @staticmethod
    def _pos(pos, hw):
        return F.interpolate(pos, size=hw, mode='bicubic', align_corners=False).flatten(2).permute(2, 0, 1)

    def _cond(self, tc):
        ft, fr, v = 30 * 2 ** tc[:, 0], 30 * 2 ** tc[:, 1], tc[:, 3].clamp_min(0)
        return torch.cat([tc[:, :3], self.psy(ft, fr, v)], 1)

    def _temporal_tokens(self, ct, cr, pairs, i, pos, trmap, film):
        """ct, cr [B, M, C, h, w] clip features of one depth -> temporal tokens [h * w, B, d]: mean over the pairs of the test change,
        the reference change and their mismatch, gated, projected, FiLM-modulated."""
        cf = self.cfanet
        bi = torch.arange(ct.shape[0], device=ct.device)[:, None]
        a, b = pairs[..., 0], pairs[..., 1]
        dT, dR = ct[bi, b] - ct[bi, a], cr[bi, b] - cr[bi, a]                                        # [B, P, C, h, w]
        d = cf.dist_func(dT, dR).mean(1)
        feat = torch.cat([cf.dist_func(dT, 0).mean(1), cf.dist_func(dR, 0).mean(1), d], 1) * self.t_gate[i](d, trmap)
        t = self.t_dim_reduce[i](feat)
        t = t * (1 + film[:, 0, :, None, None].to(t.dtype)) + film[:, 1, :, None, None].to(t.dtype)
        return t.flatten(2).permute(2, 0, 1) + self._pos(pos, feat.shape[2:]) + self.t_type

    def forward(self, test, ref, band, wins, ppd, clip=None):
        cf = self.cfanet
        B = test.shape[0]
        ppd = torch.as_tensor(ppd, device=test.device, dtype=torch.float32).reshape(-1).expand(B)
        cvmap = self.cv_enc(wins, ppd, test.shape[-2] // 16, test.shape[-1] // 16)
        cv = band.float()
        band_tok = (self.cv_band_proj(cv) + self.cv_band_pos(torch.arange(cv.shape[1], device=cv.device)).unsqueeze(0)
                    + self.cv_type).transpose(0, 1)                                                    # [nb, B, d]
        fd, fr = cf.features(test), cf.features(ref)
        if clip is not None:
            M = clip['t'].shape[1]
            ct = [f.unflatten(0, (B, M)) for f in cf.features(clip['t'].flatten(0, 1))]
            cr = [f.unflatten(0, (B, M)) for f in cf.features(clip['r'].flatten(0, 1))]
            tr_ppd = torch.as_tensor(clip['tr_ppd'], device=test.device, dtype=torch.float32).reshape(-1).expand(B)
            trmap = self.tr_enc(clip['tr_win'], tr_ppd, clip['t'].shape[-2] // 16, clip['t'].shape[-1] // 16)
            film = self.t_film(self._cond(clip['tcond'].float())).view(B, len(fd), 2, -1)
        _, _, th, tw = fd[-1].shape
        pos = torch.cat((cf.h_emb.repeat(1, 1, 1, cf.w_emb.shape[3]), cf.w_emb.repeat(1, 1, cf.h_emb.shape[2], 1)), dim=1)
        toks = []
        for i in reversed(range(len(fd))):                                                             # coarse to fine
            diff = cf.dist_func(fd[i], fr[i])
            s = torch.cat([fd[i], fr[i], diff], dim=1) * cf.weight_pool[i](diff, cvmap)
            if s.shape[2] > th and s.shape[3] > tw:
                s = F.adaptive_avg_pool2d(s, (th, tw))
            s = cf.dim_reduce[i](s).flatten(2).permute(2, 0, 1) + self._pos(pos, s.shape[2:])
            if clip is not None:
                t = self._temporal_tokens(ct[i], cr[i], clip['pairs'], i, pos, trmap, film[:, i])
                s = torch.cat([s, t.to(s.dtype)], dim=0)
            toks.append(cf.sa_attn_blks[i](s))
        q = toks[0]
        for i in range(len(toks) - 1):
            q = cf.attn_blks[i](q, toks[i + 1])
        q = torch.cat([q, band_tok.to(q.dtype)], dim=0)
        return cf.score_linear(cf.attn_pool(q).mean(dim=0)).view(-1)
