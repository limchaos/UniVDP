import json
import logging
import os

import torch

import pycvvdp
from pycvvdp.video_source import video_source_array
from pycvvdp.vq_metric import vq_metric, vq_exception

from pyunivdp.univdp_net import UniVDPNet
from pyunivdp import univdp_inputs as I

DEFAULT_CHECKPOINT = 'univdp_e4gate_s42.pt'
_PKG = os.path.dirname(os.path.abspath(__file__))


def find_checkpoint(path=None):
    """Explicit path, else $UNIVDP_CHECKPOINT, else <repository>/checkpoints/, else ~/.cache/univdp/."""
    cands = [path] if path else [os.environ.get('UNIVDP_CHECKPOINT'),
                                 os.path.join(os.path.dirname(_PKG), 'checkpoints', DEFAULT_CHECKPOINT),
                                 os.path.join(os.path.expanduser('~'), '.cache', 'univdp', DEFAULT_CHECKPOINT)]
    for c in cands:
        if c and os.path.isfile(c):
            return c
    raise vq_exception(f"UniVDP checkpoint not found (looked for {[c for c in cands if c]}). Download '{DEFAULT_CHECKPOINT}' "
                       f"into the 'checkpoints' folder of the repository, set UNIVDP_CHECKPOINT, or pass --checkpoint.")


def load_network(path, device):
    """A released checkpoint ({'state_dict', 'meta'}) or a checkpoint of the training code ({'model_state_dict', 'config', ...})."""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    sd = ck.get('state_dict', ck.get('model_state_dict'))
    if sd is None:
        raise vq_exception(f'{path} is not a UniVDP checkpoint')
    net = UniVDPNet()
    net.load_state_dict(sd, strict=True)
    return net.to(device).eval(), ck.get('meta', {'source': os.path.basename(path)})


class univdp(vq_metric):
    """UniVDP: a full-reference quality metric for images and videos, SDR and HDR, at any frame rate, for a given display and
    viewing distance. See README.md.

    The score is a quality value, higher is better, on the scale of the training targets (about 0 for the strongest distortions
    of the training sets, about 1 for no visible distortion). It is not calibrated in JOD units."""

    def __init__(self, display_name='standard_4k', display_photometry=None, display_geometry=None, config_paths=[],
                 checkpoint=None, device=None, bf16=True, quiet=False, **kwargs):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.quiet = quiet
        with open(os.path.join(_PKG, 'vvdp_data', 'univdp_parameters.json')) as f:
            self.P = json.load(f)
        self.set_display_model(display_name, display_photometry, display_geometry, config_paths)
        paths = checkpoint if isinstance(checkpoint, (list, tuple)) else [checkpoint]
        self.nets, self.meta = [], []
        for p in paths:
            net, meta = load_network(find_checkpoint(p), self.device)
            self.nets.append(net); self.meta.append(meta)
        self.bf16 = bf16 and self.device.type == 'cuda'                 # the paper's numbers: bf16 autocast on the GPU
        self.stats = I.CvvdpStatistics(self.device)

    def set_display_model(self, display_name='standard_4k', display_photometry=None, display_geometry=None, config_paths=[]):
        super().set_display_model(display_name, display_photometry, display_geometry, config_paths)
        self.config_paths = config_paths
        self.display_geometry = display_geometry if display_geometry is not None else \
            pycvvdp.vvdp_display_geometry.load(display_name, config_paths)
        self.pix_per_deg = self.display_geometry.get_ppd()

    def predict(self, test_cont, reference_cont, dim_order='BCFHW', frames_per_second=0):
        """Arrays or tensors (see pycvvdp's vq_metric.predict). frames_per_second = 0 for images."""
        vs = video_source_array(test_cont, reference_cont, frames_per_second, dim_order=dim_order,
                                display_photometry=self.display_photometry, config_paths=self.config_paths)
        return self.predict_video_source(vs, reopen=lambda: vs)

    def predict_file(self, test_fname, reference_fname, temp_resample=False, max_resample_fps=None, frames=-1, fps=None,
                     full_screen_resize=None, ffmpeg_cc=False, verbose=False):
        """Image or video files (any format pycvvdp reads)."""
        def open_source():
            if temp_resample:
                if max_resample_fps:
                    pycvvdp.video_source_temp_resample_file.max_fps = max_resample_fps
                return pycvvdp.video_source_temp_resample_file(test_fname, reference_fname, display_photometry=self.display_photometry,
                                                               config_paths=self.config_paths, full_screen_resize=full_screen_resize,
                                                               resize_resolution=self.display_geometry.resolution, frames=frames,
                                                               ffmpeg_cc=ffmpeg_cc, verbose=verbose)
            return pycvvdp.video_source_file(test_fname, reference_fname, display_photometry=self.display_photometry,
                                             config_paths=self.config_paths, full_screen_resize=full_screen_resize,
                                             resize_resolution=self.display_geometry.resolution, frames=frames, fps=fps,
                                             ffmpeg_cc=ffmpeg_cc, verbose=verbose)
        return self.predict_video_source(open_source(), reopen=open_source)

    @torch.no_grad()
    def predict_video_source(self, vid_source, reopen=None, frame_padding=None):
        """vid_source: a pycvvdp video source. Videos from files are read sequentially and need `reopen`, a function that returns
        a fresh copy of the source (predict_file passes one). Returns (score, stats)."""
        H, W, N = vid_source.get_video_size()
        if vid_source.get_batch_size() != 1:
            raise vq_exception('UniVDP processes one test / reference pair at a time (batch size 1)')
        dp, ppd = self.display_photometry, self.pix_per_deg
        o = getattr(vid_source, 'vs', vid_source)
        random_access = isinstance(o, video_source_array) or N == 1
        if N == 1:
            pos = I.image_inputs(vid_source, dp, ppd, self.stats, self.device, self.P)
            info = {'width': W, 'height': H, 'N_frames': 1, 'frames_per_second': 0}
        else:
            if reopen is None and not random_access:
                raise vq_exception('predict_video_source needs reopen= for sequential (file) video sources; use predict_file')
            first = [vid_source]
            open_source = (lambda: vid_source) if random_access else (lambda: first.pop() if first else reopen())
            pos, info = I.video_inputs(open_source, dp, ppd, self.stats, self.device, self.P, random_access=random_access)
        scores = []
        for p in pos:
            s = [float(self._forward(net, p)) for net in self.nets]
            scores.append(sum(s) / len(s))
        Q = torch.tensor(sum(scores) / len(scores))
        stats = dict(info, position_scores=scores, pix_per_deg=ppd)
        if N > 1:
            stats.update(position_frames=[p['frame'] for p in pos], position_times_s=[p['time_s'] for p in pos],
                         speed_deg_per_s=[p['speed_deg_per_s'] for p in pos])
        return Q, stats

    def _forward(self, net, p):
        dev = self.device
        clip = None
        if 'clip' in p:
            c = p['clip']
            clip = {'t': c['t'][None].to(dev), 'r': c['r'][None].to(dev), 'pairs': c['pairs'][None].to(dev),
                    'tcond': c['tcond'][None].to(dev), 'tr_win': [w[None].to(dev) for w in c['tr_win']],
                    'tr_ppd': torch.tensor([c['tr_ppd']], device=dev)}
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=self.bf16):
            q = net(p['test'][None].to(dev), p['ref'][None].to(dev), p['band'][None].to(dev),
                    [w[None].to(dev) for w in p['wins']], torch.tensor([p['ppd']], device=dev), clip=clip)
        return q.float().mean()

    def short_name(self):
        return 'UniVDP'

    def quality_unit(self):
        return 'Q'

    def get_info_string(self):
        ck = ', '.join(f"{m.get('model', '?')} seed {m.get('seed', '?')}" for m in self.meta)
        return f'UniVDP v0.1 ({ck}), display: {self.display_name}, {self.pix_per_deg:.1f} pix/deg'
