"""Inputs of the UniVDP network, computed from a pycvvdp video source.

Images: the whole frame (short side scaled to [384, 1080], never upsampled above the native size unless it is below 384) and the
ColorVideoVDP window statistics of the native frame.

Videos: K = 8 positions spread uniformly over the whole video (every second position of a 16-position grid). At each position:
  keyframe   display-encoded test and reference frame, long side capped at 1920 and stored as 8 bits, then short side min(1080, its
             short side); ColorVideoVDP window statistics of the native frame (visibility map and band tokens);
  clip       250 ms of real time starting at the keyframe, short side 224, with 4 pairs of adjacent source frames at evenly spaced
             times (frame differences stay at the native frame rate);
  motion     reference speed v = mean over the pairs of the 90th percentile of the Farneback flow magnitude (deg/s);
             frame rates of the test and reference files;
  transient  ColorVideoVDP in video mode at native resolution over the clip span, started one temporal-filter length (250 ms)
             earlier so the filters are settled; mean and max over the clip span of the transient-channel window statistics.
The video score is the mean of the 8 position scores.

pycvvdp file sources are sequential (no random access), so a video is read in two passes, each from a freshly opened source:
keyframes and clips, then the transient windows. HDR keyframes (PQ, HLG or linear EOTF) need a second source in the first pass,
because their display-encoded frames (PU21) cannot be re-wrapped with the display model.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

HDR_EOTF = ('PQ', 'HLG', 'linear')


def frame_ids(N, K):
    return sorted({int(round(i * (N - 1) / max(K - 1, 1))) for i in range(min(K, N))})


def clip_frames(t, N, fps, dur_ms, pairs):
    """Source frames of the clip that starts at frame t, and the pairs of adjacent frames as indices into that list."""
    n = min(N, max(2, int(round(dur_ms / 1000 * fps))))
    start = max(0, min(t, N - n))
    st = [int(x) for x in np.round(np.linspace(start, start + n - 2, pairs))]
    fl = sorted(set(st) | {x + 1 for x in st})
    return fl, [(fl.index(x), fl.index(x + 1)) for x in st]


def dino_input(x, short):
    """[3, H, W] in [0, 1] -> [1, 3, h, w]: aspect kept, short side `short`, both sides multiples of 16."""
    _, H, W = x.shape
    s = short / min(H, W)
    h, w = max(16, int(round(H * s / 16)) * 16), max(16, int(round(W * s / 16)) * 16)
    if (h, w) != (H, W):
        x = F.interpolate(x.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False, antialias=s < 1)[0]
    return x.clamp(0, 1).unsqueeze(0)


def cap_u8(x, cap):
    """[3, H, W] in [0, 1] -> uint8 with the long side at most `cap` (the keyframe store of the training and evaluation code)."""
    _, H, W = x.shape
    s = min(1.0, cap / max(H, W))
    if s < 1:
        x = F.interpolate(x.unsqueeze(0), size=(int(round(H * s)), int(round(W * s))), mode='bilinear', align_corners=False,
                          antialias=True)[0]
    return (x.clamp(0, 1) * 255).round().to(torch.uint8).cpu()


def _chw(x):
    x = x.squeeze(2).clamp(0, 1)
    return x[0] if x.dim() == 4 else x


class FrameView:
    """A single frame of a source as an image source (for ColorVideoVDP's image mode)."""
    def __init__(s, vs, ff): s.vs, s.ff = vs, ff
    def get_batch_size(s): return 1
    def get_frames_per_second(s): return 0
    def get_video_size(s): H, W, _ = s.vs.get_video_size(); return (H, W, 1)
    def get_test_frame(s, fr, device, colorspace): return s.vs.get_test_frame(s.ff, device=device, colorspace=colorspace)
    def get_reference_frame(s, fr, device, colorspace): return s.vs.get_reference_frame(s.ff, device=device, colorspace=colorspace)


class WindowView:
    """Frames [a, a + n) of a sequential source as a video of n frames."""
    def __init__(s, vs, a, n, fps): s.vs, s.a, s.n, s.fps = vs, a, n, fps
    def get_batch_size(s): return 1
    def get_frames_per_second(s): return s.fps
    def get_video_size(s): H, W, _ = s.vs.get_video_size(); return (H, W, s.n)
    def get_test_frame(s, i, device, colorspace): return s.vs.get_test_frame(s.a + i, device=device, colorspace=colorspace)
    def get_reference_frame(s, i, device, colorspace): return s.vs.get_reference_frame(s.a + i, device=device, colorspace=colorspace)


def _is_resampling(o):
    import pycvvdp
    return isinstance(o, pycvvdp.video_source_temp_resample_file)


def advance(vs, ff, dev, colorspace='display_encoded_01'):
    """Skip frame ff of a sequential source. Plain video files: pull the raw frame from both decoders (no colour conversion).
    Temporal-resampling sources cache the last decoded source frame without its colour space, so they are read in the colour space
    of the next real request (`colorspace`). Other sources: read as luminance."""
    o = getattr(vs, 'vs', vs)
    if type(o).__name__ in ('video_source_video_file', 'video_source_video_file_preload') and hasattr(o, 'init_readers'):
        o.init_readers()
        if ff == o.test_vidr.curr_frame + 1 and ff == o.reference_vidr.curr_frame + 1:
            o.test_vidr.get_frame(); o.reference_vidr.get_frame()
            return
    cs = colorspace if _is_resampling(o) else 'Y'
    vs.get_test_frame(ff, device=dev, colorspace=cs); vs.get_reference_frame(ff, device=dev, colorspace=cs)


def native_fps(vs):
    """(test, reference) frame rates of the files (the readers' average rate), else the source's rate."""
    o = getattr(vs, 'vs', vs)
    if hasattr(o, 'init_readers'):
        try:
            o.init_readers()
        except Exception:
            pass
    src = float(vs.get_frames_per_second() or 0)
    ft = getattr(getattr(o, 'test_vidr', None), 'avg_fps', None)
    fr = getattr(getattr(o, 'reference_vidr', None), 'avg_fps', None)
    return float(ft or src), float(fr or src)


def luma_u8(x):                                                  # [3, h, w] uint8 -> uint8 luma
    x = x.float()
    return (0.2126 * x[0] + 0.7152 * x[1] + 0.0722 * x[2]).round().clamp(0, 255).byte().numpy()


def clip_speed(r, pairs, H, fps, ppd, q=0.9):
    """Reference motion speed (deg/s) of a clip: Farneback flow between the reference frames of each pair, `q` quantile of the
    magnitude in clip pixels -> native pixels -> deg/s; mean over the pairs."""
    import cv2
    sc = H / r.shape[-2]
    sp = []
    for i, j in pairs:
        if i == j:
            continue
        flow = cv2.calcOpticalFlowFarneback(luma_u8(r[i]), luma_u8(r[j]), None, 0.5, 3, 15, 3, 5, 1.1, 0)
        sp.append(float(np.quantile(np.sqrt((flow ** 2).sum(-1)), q)) * sc * fps / ppd)
    return float(np.mean(sp))


class CvvdpStatistics:
    """ColorVideoVDP (the cvvdp-ml-saliency configuration of pycvvdp) as a feature extractor: per band and window of about one
    visual degree, the mean and variance of |T|S, |R|S and the masked difference D, for every visual channel."""

    def __init__(self, device):
        from pycvvdp.cvvdp_ml_metric import cvvdp_ml_saliency
        self.m = cvvdp_ml_saliency(quiet=True, device=device, temp_padding='replicate')
        self.dev = device

    @torch.no_grad()
    def image(self, kvs, display_photometry, ppd):
        """Image mode -> (band tokens [nb, 36] = window mean and standard deviation of the 18 statistics, windows [[Hb, Wb, 18]])."""
        m = self.m
        m.display_photometry = display_photometry; m.pix_per_deg = ppd; m.lpyr = None
        feats, _ = m.extract_features(kvs)
        band, wins = [], []
        for f in feats:
            f = f[None] if f.dim() == 5 else f
            x = f[0, 0].reshape(-1, f.shape[-2] * f.shape[-1]).float()
            band.append(torch.cat([x.mean(0), x.std(0, unbiased=False)]))
            wins.append(f[0, 0].reshape(f.shape[-4], f.shape[-3], -1).float().cpu())
        return torch.stack(band).float().cpu(), wins

    @torch.no_grad()
    def video(self, kvs, display_photometry, ppd):
        """Video mode -> per band [1, n, Hb, Wb, 4 channels, 6 statistics]."""
        m = self.m
        m.display_photometry = display_photometry; m.pix_per_deg = ppd; m.lpyr = None
        feats, _ = m.extract_features(kvs)
        return feats

    def metric_colorspace(self):
        return 'logLMS_DKLd65' if self.m.contrast == 'log' else 'DKLd65'


@torch.no_grad()
def image_inputs(vs, dp, ppd, stats, dev, P):
    t = _chw(vs.get_test_frame(0, device=dev, colorspace='display_encoded_01'))
    r = _chw(vs.get_reference_frame(0, device=dev, colorspace='display_encoded_01'))
    band, wins = stats.image(FrameView(vs, 0), dp, ppd)
    s = min(P['keyframe_short_side'], max(P['image_min_short_side'], min(t.shape[1:])))
    return [{'test': dino_input(t, s)[0], 'ref': dino_input(r, s)[0], 'band': band, 'wins': wins, 'ppd': float(ppd)}]


@torch.no_grad()
def video_inputs(open_source, dp, ppd, stats, dev, P, random_access=False):
    """Inputs of the scored positions of a video. open_source() must return a fresh source positioned at frame 0 (or the same
    source when it supports random access)."""
    from pycvvdp.video_source import video_source_array
    vs = open_source()
    H, W, N = vs.get_video_size()
    fps = float(vs.get_frames_per_second())
    ft, fr = native_fps(vs)
    ids = frame_ids(N, P['position_grid'])
    keys = list(range(0, len(ids), 2))                                       # every second position of the grid
    clips = {k: clip_frames(ids[k], N, fps, P['clip_ms'], P['clip_pairs']) for k in keys}
    key_frames = {ids[k] for k in keys}
    clip_need = {f for fl, _ in clips.values() for f in fl}
    need = key_frames | clip_need
    hdr = getattr(dp, 'EOTF', 'sRGB') in HDR_EOTF
    vs2 = (vs if random_access else open_source()) if hdr else None
    skip = (lambda s, ff: None) if random_access else (lambda s, ff: advance(s, ff, dev))
    key, frames = {}, {}
    s_clip = min(P['clip_short_side'], H, W)
    # ---- pass 1: keyframes (DINO input + window statistics) and clip frames, display-encoded
    for ff in range(max(need) + 1):
        if ff not in need:
            skip(vs, ff)
            if hdr: skip(vs2, ff)
            continue
        T0 = vs.get_test_frame(ff, device=dev, colorspace='display_encoded_01')
        R0 = vs.get_reference_frame(ff, device=dev, colorspace='display_encoded_01')
        if ff in key_frames:
            if hdr:
                band, wins = stats.image(FrameView(vs2, ff), dp, ppd)
            else:                                                            # SDR code values: re-wrap with the display model
                band, wins = stats.image(FrameView(video_source_array(T0.clamp(0, 1), R0.clamp(0, 1), 0, dim_order='BCFHW',
                                                                      display_photometry=dp), 0), dp, ppd)
            key[ff] = (cap_u8(_chw(T0), P['keyframe_cap_long_side']), cap_u8(_chw(R0), P['keyframe_cap_long_side']), band, wins)
        elif hdr:
            skip(vs2, ff)
        if ff in clip_need:
            to_u8 = lambda x: (dino_input(_chw(x), s_clip)[0] * 255).round().to(torch.uint8).cpu()
            frames[ff] = (to_u8(T0), to_u8(R0))
    # ---- pass 2: transient channel over the clip spans, at native resolution and frame rate
    vs = open_source()
    cs = stats.metric_colorspace()
    L = int(math.ceil(P['transient_warmup_ms'] / 1000 * fps / 2) * 2 + 1)
    spans = sorted((max(0, min(clips[k][0]) - (L - 1)), min(clips[k][0]), max(clips[k][0]), k) for k in keys)
    runs = []                                                                # merge overlapping windows (sequential decoding)
    for w0, s0, s1, k in spans:
        if runs and w0 <= runs[-1][1]:
            runs[-1][1] = max(runs[-1][1], s1); runs[-1][2].append((s0, s1, k))
        else:
            runs.append([w0, s1, [(s0, s1, k)]])
    tr, pos = {}, 0
    for w0, w1, members in runs:
        if not random_access:
            for ff in range(pos, w0):
                advance(vs, ff, dev, cs)
        feats = stats.video(WindowView(vs, w0, w1 - w0 + 1, fps), dp, ppd)
        pos = w1 + 1
        for s0, s1, k in members:
            tr[k] = []
            for f in feats:                                                  # [1, n, Hb, Wb, 4 channels, 6 statistics]
                x = f[0, s0 - w0:s1 - w0 + 1, :, :, 3, :].float()           # transient channel over the clip span
                tr[k].append(torch.cat([x.mean(0), x.amax(0)], -1).float().cpu())
    # ---- assemble
    out = []
    for k in keys:
        fl, pr = clips[k]
        t8, r8, band, wins = key[ids[k]]
        ct = torch.stack([frames[f][0] for f in fl]); cr = torch.stack([frames[f][1] for f in fl])
        v = clip_speed(cr, pr, H, fps, ppd, P['speed_quantile'])
        tc = [math.log2(ft / 30), math.log2(fr / 30), math.log2((1000.0 / fps) / (1000 / 30)), v]
        out.append({'test': dino_input(t8.float() / 255, min(P['keyframe_short_side'], min(t8.shape[1:])))[0],
                    'ref': dino_input(r8.float() / 255, min(P['keyframe_short_side'], min(r8.shape[1:])))[0],
                    'band': band, 'wins': wins, 'ppd': float(ppd),
                    'clip': {'t': ct.float() / 255, 'r': cr.float() / 255, 'pairs': torch.tensor(pr, dtype=torch.long),
                             'tcond': torch.tensor(tc), 'tr_win': tr[k], 'tr_ppd': float(ppd)},
                    'frame': ids[k], 'time_s': ids[k] / fps, 'speed_deg_per_s': v})
    info = {'width': W, 'height': H, 'N_frames': N, 'frames_per_second': fps, 'fps_test': ft, 'fps_ref': fr}
    return out, info
