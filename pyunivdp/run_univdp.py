# Command-line interface for UniVDP (the options follow ColorVideoVDP's `cvvdp` command).
import argparse
import glob
import logging
import shlex
import sys
import traceback

import torch

import pycvvdp
from pycvvdp.vq_metric import vq_exception

from pyunivdp import univdp


def expand_wildcards(filestrs):
    files = []
    for f in filestrs:
        files += sorted(glob.glob(f)) if '*' in f else [f]
    return files


def parse_args(arg_list=None):
    p = argparse.ArgumentParser(description='Evaluate UniVDP on test / reference images or videos')
    p.add_argument('-t', '--test', type=str, nargs='+', help='list of test images/videos')
    p.add_argument('-r', '--ref', type=str, nargs='+', help='list of reference images/videos')
    p.add_argument('-d', '--display', type=str, default='standard_4k', help="display name, e.g. 'standard_fhd', or ? to print the list of models (ColorVideoVDP display models).")
    p.add_argument('-c', '--config-paths', type=str, nargs='+', default=[], help='paths to ColorVideoVDP configuration files or directories, e.g. your own display_models.json')
    p.add_argument('--checkpoint', type=str, nargs='+', default=None, help='UniVDP checkpoint(s); several checkpoints are averaged. Default: checkpoints/univdp_e4gate_s42.pt or $UNIVDP_CHECKPOINT')
    p.add_argument('--device', type=str, default='cuda', help="PyTorch device: 'cpu', 'cuda', 'cuda:0', ...")
    p.add_argument('--pix-per-deg', type=float, default=None, help='overwrite the display geometry with this number of pixels per visual degree')
    p.add_argument('--temp-resample', type=float, nargs='?', default=-1, const=0, help='resample test and reference to a common frame rate (test and reference of different frame rates); optional argument: the maximum frame rate')
    p.add_argument('-f', '--full-screen-resize', choices=['bilinear', 'bicubic', 'nearest', 'area'], default=None, help='resize test and reference videos to the full resolution of the display')
    p.add_argument('-n', '--nframes', type=int, default=-1, help='the number of video frames to compare')
    p.add_argument('--count-frames', action='store_true', help='count the frames of the videos exactly (slower); use if you see frame read errors')
    p.add_argument('--fps', type=float, default=None, help='frame rate, overwrites the rate stored in the video file')
    p.add_argument('--ffmpeg-cc', action='store_true', help='use ffmpeg for upsampling and colour conversion')
    p.add_argument('--fp32', action='store_true', help='run the network in float32 on the GPU (the paper used bfloat16 autocast)')
    p.add_argument('--details', action='store_true', help='print the score of every scored position of a video')
    p.add_argument('--result', type=str, default=None, help='write the results to this CSV file')
    p.add_argument('-q', '--quiet', action='store_true', help='print only the scores')
    p.add_argument('-v', '--verbose', action='store_true', help='print extra information')
    p.add_argument('--debug', action='store_true', help='print the full stack trace on errors')
    p.add_argument('-i', '--interactive', action='store_true', help='read the arguments of each run from the standard input, line by line (the model is loaded once)')
    return p.parse_args(arg_list)


_METRIC = {}


def get_metric(args, device, dp, dg):
    key = (args.display, tuple(args.config_paths), args.pix_per_deg, tuple(args.checkpoint or []), str(device), args.fp32)
    if key not in _METRIC:
        _METRIC.clear()
        _METRIC[key] = univdp(display_name=args.display, display_photometry=dp, display_geometry=dg, config_paths=args.config_paths,
                              checkpoint=args.checkpoint, device=device, bf16=not args.fp32, quiet=args.quiet)
        _METRIC[key].display_name = args.display
    return _METRIC[key]


def run_on_args(args):
    logging.basicConfig(format='[%(levelname)s] %(message)s',
                        level=logging.ERROR if args.quiet else (logging.DEBUG if args.verbose else logging.INFO), force=True)
    if args.display == '?':
        pycvvdp.vvdp_display_photometry.list_displays(args.config_paths)
        return
    if args.test is None or args.ref is None:
        logging.error('Paths to both test and reference content need to be specified.')
        return
    device = torch.device(args.device if not args.device.startswith('cuda') or torch.cuda.is_available() else 'cpu')
    if str(device) != args.device:
        logging.warning(f'The requested device ({args.device}) is not available, running on the CPU. This may be slow.')
    tests, refs = expand_wildcards(args.test), expand_wildcards(args.ref)
    if not tests or not refs or (len(tests) != len(refs) and len(tests) != 1 and len(refs) != 1):
        logging.error('Pass the same number of test and reference sources, or a single reference, or a single test.')
        return
    dp = pycvvdp.vvdp_display_photometry.load(args.display, config_paths=args.config_paths)
    dg = pycvvdp.vvdp_display_geometry.load(args.display, config_paths=args.config_paths) if args.pix_per_deg is None else \
        pycvvdp.vvdp_display_geometry([1024, 1024], ppd=args.pix_per_deg)
    metric = get_metric(args, device, dp, dg)
    logging.info('When reporting metric results, please include the following information:')
    logging.info(metric.get_info_string())
    res = open(args.result, 'w') if args.result else None
    if res:
        res.write('test, reference, UniVDP\n')
    for k in range(max(len(tests), len(refs))):
        tf, rf = tests[min(k, len(tests) - 1)], refs[min(k, len(refs) - 1)]
        logging.info(f"Predicting the quality of '{tf}' compared to '{rf}'")
        Q, stats = metric.predict_file(tf, rf, temp_resample=args.temp_resample >= 0, max_resample_fps=args.temp_resample or None,
                                       frames=-2 if args.count_frames else args.nframes, fps=args.fps,
                                       full_screen_resize=args.full_screen_resize, ffmpeg_cc=args.ffmpeg_cc, verbose=args.verbose)
        print(f'{Q.item():0.4f}' if args.quiet else f'UniVDP={Q.item():0.4f} [Q]')
        if args.details and stats['N_frames'] > 1:
            for f, t, v, s in zip(stats['position_frames'], stats['position_times_s'], stats['speed_deg_per_s'], stats['position_scores']):
                print(f'  frame {f:6d}  t = {t:7.3f} s  v = {v:7.2f} deg/s  score {s:0.4f}')
        if res:
            res.write(f'{tf}, {rf}, {Q.item()}\n')
    if res:
        res.close()


def main():
    args = parse_args()
    try:
        if args.interactive:
            for line in sys.stdin:
                run_on_args(parse_args(shlex.split(line)))
        else:
            run_on_args(args)
    except vq_exception as ex:
        logging.error(str(ex))
        if args.debug:
            traceback.print_exc()


if __name__ == '__main__':
    main()
