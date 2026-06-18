import multiprocessing as mp
import ffmpeg

import torch
import torchvision
import numpy as np
from tqdm import tqdm
from munch import Munch

from exp import ex
from ckpt import load_ckpt_adapt
from model import get_pos_neg_extractors
from optimizer import get_optimizer
from utils import prepare_batch, get_all

from metrics.log import Logger, Metric
from metrics.wild360 import calc_score, visualize_heatmap, get_gt_heatmap


@ex.capture()
def demo_adapt(log_path, config_dir, max_epoch, lr, clip_length, num_workers,
         display_config, num_frame, eval_start_epoch, save_model,
         input_video, extractor_name, feature_norm,
         model_config_pos, model_config_neg, cache_path, device):
    model_config_pos = Munch(model_config_pos)
    model_config_neg = Munch(model_config_neg)

    probe = ffmpeg.probe(input_video)
    video_stream = next((stream for stream in probe['streams']
                         if stream['codec_type'] == 'video'), None)
    orig_width = int(video_stream['width'])
    orig_height = int(video_stream['height'])

    width = model_config_pos.input_resolution * 2
    height = model_config_pos.input_resolution

    cmd = ffmpeg.input(input_video).filter('scale', width, height)
    out, _ = (
        cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
           .run(capture_stdout=True, quiet=True)
    )
    video = np.frombuffer(out, np.uint8)
    video = video.reshape([-1, height, width, 3])
    video = torch.from_numpy(video.astype('float32')).permute(0, 3, 1, 2)

    if feature_norm:
        video = ((video / 255.) - 0.5) / 0.5
    
    device = torch.device(device)

    # Run pos/neg extractors (matching VQA-ODV pipeline)
    extractors = get_pos_neg_extractors(extractor_name=extractor_name,
                                        model_config_pos=model_config_pos,
                                        model_config_neg=model_config_neg,
                                        cache_path=cache_path)
    feature_extractor_pos = extractors["pos_extractor"].cuda()
    feature_extractor_neg = extractors["neg_extractor"].cuda()
    feature_extractor_pos.eval()
    feature_extractor_neg.eval()

    with torch.no_grad():
        video_cuda = video.to(device, non_blocking=True)
        frame_pos = feature_extractor_pos(video_cuda).detach().cpu()
        frame_neg = feature_extractor_neg(video_cuda).detach().cpu()
        del video_cuda

    # Pad to clip_length if needed
    if frame_pos.size(0) < clip_length:
        zero_pos = torch.zeros_like(frame_pos[0])
        zero_neg = torch.zeros_like(frame_neg[0])
        if len(frame_pos.size()) == 3:
            zero_pos = zero_pos.repeat(clip_length - frame_pos.size(0), 1, 1)
            zero_neg = zero_neg.repeat(clip_length - frame_neg.size(0), 1, 1)
        else:
            zero_pos = zero_pos.repeat(clip_length - frame_pos.size(0), 1, 1, 1)
            zero_neg = zero_neg.repeat(clip_length - frame_neg.size(0), 1, 1, 1)
        frame_pos = torch.cat((frame_pos, zero_pos), 0)
        frame_neg = torch.cat((frame_neg, zero_neg), 0)
    elif frame_pos.size(0) > clip_length:
        # Trim longer clips to the configured length
        frame_pos = frame_pos[:clip_length]
        frame_neg = frame_neg[:clip_length]

    if len(frame_pos.size()) == 3:
        cls_pos = frame_pos[:, 0]
        cls_neg = frame_neg[:, 0]
        frame_pos = frame_pos[:, 1:]
        frame_neg = frame_neg[:, 1:]
    else:
        cls_pos = frame_pos[:, 0]
        cls_neg = frame_neg[:, 0]

    # Build mask like VQA-ODV __getitem__
    mask = torch.where(torch.norm(frame_pos, dim=(-2, -1)) > 1e-6, 1., 0.)
    if len(frame_pos.size()) > 3:
        mask = torch.where(torch.norm(frame_pos, dim=(-2, -1)).sum(-1) > 1e-5, 1., 0.)

    # Run inference
    dataloaders, model = get_all(device=device, modes=[])
    display_config = Munch(display_config)
    # model = load_ckpt_adapt(model)
    model.eval()

    data = {
        'frame_pos': frame_pos.unsqueeze(0).to(device),
        'frame_neg': frame_neg.unsqueeze(0).to(device),
        'cls_pos': cls_pos.unsqueeze(0).to(device),
        'cls_neg': cls_neg.unsqueeze(0).to(device),
    }
    label = {'mask': mask.unsqueeze(0).to(device)}

    result = model(data, label)
    result['heatmap'] = model.compute_heatmap(result['output'].contiguous())
    vis = torch.cat([visualize_heatmap(result['heatmap'][0][j], overlay=False).unsqueeze(0)
                     for j in range(clip_length)]).unsqueeze(0)

    iid = input_video.split('/')[-1][:-4]
    torchvision.io.write_video(f'./qual/n27/{iid}_out_imgn_011.mp4',
                               vis.squeeze(0).permute(0, 2, 3, 1), fps=clip_length)

    return 0
