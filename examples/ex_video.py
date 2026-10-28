# Score video files, or a synthetic panning video with temporal noise.
# Usage: python examples/ex_video.py <test video> <reference video> [display]
#        python examples/ex_video.py --synthetic <image file>
import sys

import imageio.v2 as imageio
import numpy as np

import pyunivdp

if sys.argv[1] != '--synthetic':
    metric = pyunivdp.univdp(display_name=sys.argv[3] if len(sys.argv) > 3 else 'standard_4k')
    Q, stats = metric.predict_file(sys.argv[1], sys.argv[2])
    print(f'UniVDP = {Q.item():.4f}')
    for f, t, v, s in zip(stats['position_frames'], stats['position_times_s'], stats['speed_deg_per_s'], stats['position_scores']):
        print(f'  frame {f:5d}  t = {t:6.2f} s  reference speed {v:6.2f} deg/s  score {s:.4f}')
    sys.exit()

# A 1 s, 60 Hz reference that pans over the image at 4 px per frame, and test videos with temporal noise (a different noise
# pattern in every frame) of increasing amplitude.
img = imageio.imread(sys.argv[2])[..., :3].astype(np.float32) / 255
h, w = min(img.shape[0], 360), min(img.shape[1], 640)
fps, N, step = 60, 60, 4
pad = np.concatenate([img, img], 1)
ref = np.stack([pad[:h, (k * step) % img.shape[1]:(k * step) % img.shape[1] + w] for k in range(N)])   # [F, H, W, C] in [0, 1]
metric = pyunivdp.univdp(display_name='standard_fhd')
rng = np.random.default_rng(0)
for sigma in (0.0, 0.02, 0.05):
    test = np.clip(ref + rng.normal(0, sigma, ref.shape).astype(np.float32), 0, 1)
    Q, _ = metric.predict(test, ref, dim_order='FHWC', frames_per_second=fps)
    print(f'temporal noise sigma {sigma:.2f}: UniVDP = {Q.item():.4f}')
