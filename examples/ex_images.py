# Score an image distorted with increasing levels of noise and blur.
# Usage: python examples/ex_images.py <image file>
import sys

import imageio.v2 as imageio
import numpy as np
import torch

import pyunivdp

ref = imageio.imread(sys.argv[1])[..., :3].astype(np.float32) / 255           # display-encoded RGB in [0, 1]
metric = pyunivdp.univdp(display_name='standard_fhd')
print(metric.get_info_string())

rng = np.random.default_rng(0)
for sigma in (0.0, 0.02, 0.05, 0.1):
    test = np.clip(ref + rng.normal(0, sigma, ref.shape).astype(np.float32), 0, 1)
    Q, _ = metric.predict(test, ref, dim_order='HWC')
    print(f'noise sigma {sigma:.2f}: UniVDP = {Q.item():.4f}')

for size in (3, 7, 15):
    x = torch.from_numpy(ref).permute(2, 0, 1)[None]
    test = torch.nn.functional.avg_pool2d(x, size, stride=1, padding=size // 2, count_include_pad=False)[0].permute(1, 2, 0).numpy()
    Q, _ = metric.predict(test, ref, dim_order='HWC')
    print(f'box blur {size}x{size}: UniVDP = {Q.item():.4f}')
