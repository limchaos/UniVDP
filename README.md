# UniVDP: a full-reference quality metric for images and videos

UniVDP is a full-reference visual quality metric that predicts the perceived quality of a distorted image or video compared with its reference. It is a successor of [ColorVideoVDP](https://github.com/gfxdisp/ColorVideoVDP): it keeps the psychophysical model of ColorVideoVDP (display model, castleCSF, contrast masking, the transient channel) and combines it with features of a fine-tuned DINOv3 vision transformer and a model of temporal change.

The main features:
* one model for images and videos, SDR and HDR (PQ, HLG, linear), and any frame rate, including test and reference videos of different frame rates;
* accounts for the display and the viewing conditions through ColorVideoVDP display models (size, resolution, viewing distance, peak luminance, ambient light);
* models the visibility of distortions (ColorVideoVDP visibility map and band statistics guide where the network pools feature differences);
* models temporal distortions: frame differences in a 250 ms clip at the native frame rate, the ColorVideoVDP transient channel, and psychophysical terms of judder, hold blur, flicker and motion masking computed from the motion speed and the frame rates.

UniVDP is implemented in PyTorch. It runs on a CUDA GPU (recommended) and on the CPU (much slower for video).

The metric is described in a paper that is currently under review. The citation will be added here.

**Table of contents**
- [Quickstart](#quickstart)
- [Checkpoint](#checkpoint)
- [Examples](#examples)
- [Display specification](#display-specification)
- [HDR content](#hdr-content)
- [Frame rates](#frame-rates)
- [Predicted quality scores](#predicted-quality-scores)
- [Python interface](#python-interface)
- [How the metric works](#how-the-metric-works)
- [Accuracy of the released checkpoint](#accuracy-of-the-released-checkpoint)
- [Run time](#run-time)
- [Training](#training)
- [Reporting metric results](#reporting-metric-results)
- [Release notes](#release-notes)
- [License and third-party code](#license-and-third-party-code)

## Quickstart

1. Create a new environment (Python 3.10 or newer; tested with 3.13):
```bash
conda create -n univdp python=3.13
conda activate univdp
```

2. Install [ffmpeg](https://ffmpeg.org/) and [FreeImage](https://freeimage.sourceforge.io/), which ColorVideoVDP uses to read videos and images:
```bash
conda install ffmpeg conda-forge::freeimage
```

3. Install PyTorch with CUDA support by following [these instructions](https://pytorch.org/get-started/locally/).

4. Clone this repository and install it. This also installs ColorVideoVDP v0.5.6 from its GitHub repository (the version UniVDP was trained with) and timm:
```bash
git clone <this repository> UniVDP
cd UniVDP
pip install -e .
```

5. Download the checkpoint `univdp_e4gate_s42.pt` (see [Checkpoint](#checkpoint)) into the `checkpoints/` folder.

Then run `univdp` from the command line:
```bash
univdp --test test_file --ref ref_file --display standard_fhd
```
The test and reference files can be images or videos. The option `--display` specifies the display on which the content is viewed, as in ColorVideoVDP.

The first run downloads a small ColorVideoVDP-ML file from the Hugging Face Hub (`gfxdisp/cvvdp_ml`), because UniVDP uses the cvvdp-ml-saliency configuration of ColorVideoVDP to compute its visibility statistics.

## Checkpoint

| File | Size | SHA256 |
|---|---|---|
| `univdp_e4gate_s42.pt` | 102 MB | `4481a5a36a25cb38ee78e4c93c592d8619491023cba539a5d85490d554c6ace9` |

Download link: TODO.

UniVDP looks for the checkpoint in this order: the `--checkpoint` option (or `checkpoint=` in Python), the environment variable `UNIVDP_CHECKPOINT`, `checkpoints/univdp_e4gate_s42.pt` in the repository, and `~/.cache/univdp/univdp_e4gate_s42.pt`. Check the file with `sha256sum checkpoints/univdp_e4gate_s42.pt`.

The checkpoint holds 25.5 M parameters, of which 14.5 M are DINOv3 ViT-S/16 blocks 0 to 7 fine-tuned with the metric. Checkpoints written by the training code (`checkpoints/last.pt` of a run in [training/](training/)) can be passed to `--checkpoint` directly.

## Examples

Compare a distorted video with its reference, viewed on a 4K display:
```bash
univdp --test test.mp4 --ref reference.mp4 --display standard_4k
```

Compare all `test-*.mp4` files with the same reference and store the results in a CSV file:
```bash
univdp --test test-*.mp4 --ref reference.mp4 --display standard_fhd --result results.csv
```

Show the score of every scored position of a video, with its time and motion speed:
```bash
univdp --test test.mp4 --ref reference.mp4 --display standard_4k --details
```

Compare an HDR video encoded with PQ:
```bash
univdp --test test_hdr.mp4 --ref reference_hdr.mp4 --display standard_hdr_pq
```

Compare a 30 Hz test video with its 120 Hz reference:
```bash
univdp --test test_30fps.mp4 --ref reference_120fps.mp4 --display standard_fhd --temp-resample
```

Compare two images on a custom display described in `my_displays/display_models_custom.json`:
```bash
univdp --test test.png --ref reference.png --config-paths my_displays --display my_display
```

Run `univdp --help` for all options. The options follow the `cvvdp` command of ColorVideoVDP where they have the same meaning. The [examples](examples/) folder shows how to call UniVDP from Python.

## Display specification

UniVDP uses the display models of ColorVideoVDP. The display model determines the colour space and transfer function of the input (SDR, PQ, HLG or linear), the luminance range, and the number of pixels per visual degree, which sets the physical frequency of every band of the visibility model and the motion speed in degrees per second. Use one of the standard displays (`standard_4k`, `standard_fhd`, `standard_hdr_pq`, `standard_hdr_hlg`, `standard_hdr_linear`), or describe your display in a `display_models_*.json` file and pass its folder with `--config-paths`. Run with `--display ?` to list the available models. The format of the file and the meaning of each field are explained in the [ColorVideoVDP documentation](https://github.com/gfxdisp/ColorVideoVDP#display-specification).

As in ColorVideoVDP, the specification is of the display, not of the content: a 1920x1080 video on `standard_4k` occupies the central quarter of a 4K display. Pass `--full-screen-resize bilinear` to upscale videos to the full display resolution. Use `--pix-per-deg` to set the viewing geometry directly.

## HDR content

HDR videos encoded with PQ or HLG need a display model with that transfer function, for example `--display standard_hdr_pq`. HDR images (OpenEXR, `.hdr`) must hold absolute linear values in cd/m^2 and need `--display standard_hdr_linear`. The network sees HDR content through the display encoding of ColorVideoVDP (PU21 for HDR), and the visibility statistics are computed from the absolute colorimetric values, as for ColorVideoVDP.

## Frame rates

The test and reference videos may have different frame rates, for example a 30 Hz test and a 120 Hz reference. Pass `--temp-resample`: both videos are then read at a common rate (frames of the slower video are repeated, which is how a display shows them). UniVDP reads the native frame rates of both files and uses them in its frame-rate and motion terms. Videos of the same frame rate need no option.

## Predicted quality scores

UniVDP reports one quality value per image or video, higher is better. The scale is that of the training targets: the JOD scores of the image training data (UPIQ), mapped linearly so that the range of the training targets spans [0, 1]. Identical test and reference score about 0.94; stronger distortions score lower, and values slightly outside [0, 1] are possible. The training loss combines a squared error on these targets with a correlation loss (Norm-in-Norm), so the scores are not calibrated in JOD units, unlike ColorVideoVDP: use them to rank or compare conditions, or fit a monotonic mapping to your own subjective scale.

## Python interface

```python
import pyunivdp

metric = pyunivdp.univdp(display_name='standard_4k')        # loads checkpoints/univdp_e4gate_s42.pt; device='cuda' if available

# Files (images or videos)
Q, stats = metric.predict_file('test.mp4', 'reference.mp4')
print(Q.item(), stats['position_scores'])

# Different frame rates
Q, stats = metric.predict_file('test_30fps.mp4', 'reference_120fps.mp4', temp_resample=True)

# Arrays: an image as [H, W, C] display-encoded values (uint8 or float in [0, 1])
Q, stats = metric.predict(test_img, ref_img, dim_order='HWC')

# Arrays: a video as [F, H, W, C] at 60 frames per second
Q, stats = metric.predict(test_vid, ref_vid, dim_order='FHWC', frames_per_second=60)
```
`metric.predict_video_source(vs, reopen=...)` takes any pycvvdp video source. Video files are read sequentially, so it needs a function that reopens the source (`predict_file` passes one). `stats` contains the per-position scores, the frame index, time and reference motion speed of each scored position, and the frame rates that were used.

## How the metric works

For an image, the whole frame is scored. Its short side is scaled to between 384 and 1080 pixels (never upsampled unless it is below 384 pixels), and ColorVideoVDP statistics are computed on the native frame.

For a video, UniVDP scores 8 positions spread uniformly over the whole video and reports the mean of their scores. At each position:
1. **Keyframe.** The test and reference frames, short side 1080 pixels (or the native size if smaller), are passed through DINOv3 ViT-S/16; features are taken after blocks 2, 4 and 7. ColorVideoVDP, run in image mode on the native frame, gives per-band statistics of the test, reference and masked difference in windows of about one visual degree.
2. **Visibility-guided pooling.** At each feature depth, a gate decides where feature differences are pooled. The gate sees the feature differences and a map encoded from the ColorVideoVDP window statistics. Per-band summaries of the statistics also enter the final attention pooling as tokens.
3. **Clip.** A 250 ms clip starting at the keyframe (short side 224 pixels) gives 4 pairs of adjacent frames at the native frame rate. The changes of the DINOv3 features between the frames of each pair, for the test and the reference, form temporal tokens. Their gate sees the ColorVideoVDP transient channel of the clip (video mode, native resolution and frame rate).
4. **Motion and frame rate.** The reference motion speed (90th percentile of the Farneback optical flow magnitude, in degrees per second) and the test and reference frame rates drive psychophysical terms of judder, hold blur, flicker and motion masking. These modulate the temporal tokens.
5. **Attention head.** Self-attention within each depth, cross-attention from coarse to fine depths, and attention pooling give the score of the position.

All sampling parameters are in [pyunivdp/vvdp_data/univdp_parameters.json](pyunivdp/vvdp_data/univdp_parameters.json). They are the values the checkpoint was trained and evaluated with; changing them requires retraining.

## Accuracy of the released checkpoint

The released checkpoint is the model of the paper's main table trained with seed 42 (last epoch of a fixed 50-epoch schedule, no checkpoint selection). Training data: LIVE, TID2013, Narwaria et al. and Korshunov et al. HDR images (via UPIQ), XR-DAVID, AVT-VQDB-UHD-1 (AVT4) and KoSMo. None of the test sets below was used for training.

Spearman rank-order correlation (|SROCC|) with the subjective scores:

| PIPAL | CSIQ | KADID-10k | LIVE-HDR | HDR-VDC | ICME | QoMEX | BVI-HFR | BVI-VFI | Mean |
|---|---|---|---|---|---|---|---|---|---|
| .709 | .963 | .924 | .924 | .825 | .762 | .929 | .876 | .679 | .843 |

HDR-VDC excludes the contents that share source material with the training data (Bonfire). The paper reports the mean over five training seeds (mean .835 over the nine sets); seed 42 is one of them. These numbers were computed with bfloat16 autocast on an NVIDIA GH200 GPU; on the CPU (float32) individual scores differ in the third decimal.

## Run time

TODO: filled in from the verification run.

## Training

[training/](training/) contains the training code, the data preparation steps and the commands to reproduce the released checkpoint. See [training/README.md](training/README.md).

## Reporting metric results

When reporting results, please include the string printed by the metric, for example:
`UniVDP v0.1 (E4-gate seed 42), display: standard_4k, 75.4 pix/deg`
and the display model used.

## Release notes

* v0.1.0 (October 2026): first release, checkpoint `univdp_e4gate_s42.pt`.

## License and third-party code

License: TODO (to be decided by the authors).

* The attention head adapts CFANet of TOPIQ (C. Chen et al., IEEE TIP 2024) from [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch), which is distributed under the S-Lab License 1.0 for non-commercial use. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
* The checkpoint contains DINOv3 weights (Meta AI) fine-tuned with the metric. Their use is subject to the DINOv3 License.
* [ColorVideoVDP](https://github.com/gfxdisp/ColorVideoVDP) is a dependency (installed with `pip`), not part of this repository.
