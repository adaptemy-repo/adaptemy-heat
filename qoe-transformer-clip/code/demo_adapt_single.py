import multiprocessing as mp
import ffmpeg

import torch
import torchvision
import numpy as np
from tqdm import tqdm
from munch import Munch

from exp import ex
from ckpt import load_ckpt_adapt
from model import get_extractor
from optimizer import get_optimizer
from utils import prepare_batch, get_all

from metrics.log import Logger, Metric
from metrics.wild360 import calc_score, visualize_heatmap, get_gt_heatmap


@ex.capture()
def demo_adapt_single(log_path, config_dir, max_epoch, lr, clip_length, num_workers,
         display_config, num_frame, eval_start_epoch, save_model,
         input_video, extractor_name, feature_norm,
         model_config, cache_path, device):
    model_config = Munch(model_config)

    probe = ffmpeg.probe(input_video)
    video_stream = next((stream for stream in probe['streams']
                         if stream['codec_type'] == 'video'), None)
    orig_width = int(video_stream['width'])
    orig_height = int(video_stream['height'])

    width = model_config.input_resolution * 2
    height = model_config.input_resolution

    cmd = ffmpeg.input(input_video).filter('scale', width, height)
    out, _ = (
        cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
           .run(capture_stdout=True, quiet=True)
    )
    video = np.frombuffer(out, np.uint8)
    video = video.reshape([-1, height, width, 3])
    video = torch.from_numpy(video.astype('float32')).permute(0, 3, 1, 2)

    if feature_norm:
        if model_config.train_type in ['dino']:
            video = video / 255.
            video -= torch.from_numpy(np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)).double()
            video /= torch.from_numpy(np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)).double()
        else:
            video = ((video / 255.) - 0.5) / 0.5
    
    device = torch.device(device)

    # Run single extractor
    feature_extractor = get_extractor(extractor_name=extractor_name,
                                      model_config=model_config,
                                      cache_path=cache_path).cuda()
    feature_extractor.eval()

    with torch.no_grad():
        video_cuda = video.to(device, non_blocking=True)
        frame = feature_extractor(video_cuda).detach().cpu()
        del video_cuda

    # Pad to clip_length if needed
    if frame.size(0) < clip_length:
        zero_vid = torch.zeros_like(frame[0])
        if len(frame.size()) == 3:
            zero_vid = zero_vid.repeat(clip_length - frame.size(0), 1, 1)
        else:
            zero_vid = zero_vid.repeat(clip_length - frame.size(0), 1, 1, 1)
        frame = torch.cat((frame, zero_vid), 0)
    elif frame.size(0) > clip_length:
        # Trim longer clips to the configured length
        frame = frame[:clip_length]

    if len(frame.size()) == 3:
        cls = frame[:, 0]
        frame = frame[:, 1:]
    else:
        cls = frame[:, 0]

    # Build mask like VQA-ODV_Single __getitem__
    mask = torch.where(torch.norm(frame, dim=(-2, -1)) > 1e-6, 1., 0.)
    if len(frame.size()) > 3:
        mask = torch.where(torch.norm(frame, dim=(-2, -1)).sum(-1) > 1e-5, 1., 0.)

    # Run inference
    dataloaders, model = get_all(device=device, modes=[])
    display_config = Munch(display_config)
    model = load_ckpt_adapt(model)
    model.eval()

    data = {
        'frame': frame.unsqueeze(0).to(device),
        'cls': cls.unsqueeze(0).to(device),
    }
    label = {'mask': mask.unsqueeze(0).to(device)}

    result = model(data, label)
    result['heatmap'] = model.compute_heatmap(result['output'].contiguous())
    vis = torch.cat([visualize_heatmap(result['heatmap'][0][j], overlay=False).unsqueeze(0)
                     for j in range(clip_length)]).unsqueeze(0)

    iid = input_video.split('/')[-1][:-4]
    torchvision.io.write_video(f'./qual_single/{iid}_out.mp4',
                               vis.squeeze(0).permute(0, 2, 3, 1), fps=clip_length)

    return 0
