import multiprocessing as mp

from tqdm import tqdm
from munch import Munch

from exp import ex
from ckpt import save_ckpt
from optimizer import get_optimizer
from utils import prepare_batch, get_all

from metrics.log import Logger, Metric
from metrics.wild360 import calc_score, visualize_heatmap, get_gt_heatmap


@ex.capture()
def train(log_path, config_dir, max_epoch, lr, clip_length, num_workers,
          display_config, num_frame, eval_start_epoch, save_model, model_name,
          use_exp_decay, use_epoch_decay, iterdecay_rate, epochdecay_rate):
    # Only build the training split to skip evaluation
    dataloaders, model = get_all(modes=['train'])
    display_config = Munch(display_config)

    logger = Logger(log_path / config_dir)
    # iters, for logging purpose
    it = 0

    optimizer, scheduler = get_optimizer(
        model=model, t_total=dataloaders['train'].dataset.t_total
    )
    optimizer.zero_grad()

    for epoch in range(max_epoch):

        model.train()

        for batch in tqdm(dataloaders['train'], desc=f"Train{epoch:02d}"):
            if model_name in ['TokenCut', 'DINO', 'LOST']:
                break

            data, label, meta = prepare_batch(batch)
            data['frame_pos_orig'] = data['frame_pos']
            data['frame_neg_orig'] = data['frame_neg']
            data['cls_pos_orig'] = data['cls_pos']
            data['cls_neg_orig'] = data['cls_neg']
            label['mask_orig'] = label['mask']
            prev_frame = None

            for i in range(0, clip_length, num_frame):
                if prev_frame is not None:
                    label['prev_frame'] = prev_frame.cuda()

                if num_frame == 1:
                    data['frame_pos'] = data['frame_pos_orig'][:, i]
                    data['frame_neg'] = data['frame_neg_orig'][:, i]
                    data['cls_pos'] = data['cls_pos_orig'][:, i]
                    data['cls_neg'] = data['cls_neg_orig'][:, i]
                    label['mask'] = label['mask_orig'][:, i]
                elif num_frame > 1:
                    data['frame_pos'] = data['frame_pos_orig'][:, i:i+num_frame]
                    data['frame_neg'] = data['frame_neg_orig'][:, i:i+num_frame]
                    data['cls_pos'] = data['cls_pos_orig'][:, i:i+num_frame]
                    data['cls_neg'] = data['cls_neg_orig'][:, i:i+num_frame]
                    label['mask'] = label['mask_orig'][:, i:i+num_frame]

                    if label['mask'].sum() == 0:
                        break
                    elif label['mask'].sum() < num_frame:
                        diff = int(num_frame - label['mask'].sum())
                        if i < diff:
                            break
                        data['frame_pos'] = data['frame_pos_orig'][:, i-diff:i+num_frame-diff]
                        data['frame_neg'] = data['frame_neg_orig'][:, i-diff:i+num_frame-diff]
                        data['cls_pos'] = data['cls_pos_orig'][:, i-diff:i+num_frame-diff]
                        data['cls_neg'] = data['cls_neg_orig'][:, i-diff:i+num_frame-diff]
                        label['mask'] = label['mask_orig'][:, i-diff:i+num_frame-diff]
                else:
                    raise ValueError("Invalid num_frame value")

                result = model(data, label)

                if result['loss_total'] == 0.:
                    continue

                result['loss_total'].backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if num_frame == 1:
                    prev_frame = result['output'].detach()

                if it % 20 == 0:
                    print(f"[train] epoch {epoch:02d} iter {it:06d} loss {result['loss_total'].item():.6f}")
                    logger.log_iter(it,
                                    optimizer.param_groups[0]['lr'],
                                    result,
                                    meta,
                                    'train')
                it += 1

                if use_exp_decay:
                    model.drop_prob *= iterdecay_rate

        if use_epoch_decay:
            model.drop_prob *= epochdecay_rate

        if save_model:
            save_ckpt(epoch, 0, model)
        logger.log_epoch(epoch, 0, {'loss': None}, {}, 'train')

    return {}
