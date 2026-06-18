import torch
from munch import Munch
from tqdm import tqdm

from exp import ex
from ckpt import save_ckpt
from optimizer import get_optimizer
from utils import prepare_batch, get_all
from metrics.log import Logger


@ex.capture()
def train(log_path, config_dir, max_epoch, lr, clip_length, num_workers, device,
          display_config, num_frame, eval_start_epoch, save_model, model_name,
          use_exp_decay, use_epoch_decay, iterdecay_rate, epochdecay_rate):
    
    
    dataloaders, model = get_all(modes=["train"])
    display_config = Munch(display_config)

    logger = Logger(log_path / config_dir)
    it = 0

    optimizer, scheduler = get_optimizer(
        model=model, t_total=dataloaders["train"].dataset.t_total
    )
    optimizer.zero_grad()

    for epoch in range(max_epoch):
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(tqdm(dataloaders["train"], desc=f"SaliVQA Train {epoch:02d}")):
            data, label, meta = prepare_batch(batch, device=device)
            result = model(data, label)
            loss = result.get("loss_total")
            if loss is None:
                raise RuntimeError("loss_total missing from model output.")

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            if step % 50 == 0:
                print(f"[train_salivqa] epoch {epoch:02d} iter {step:04d} loss {loss.item():.6f}")
                logger.log_iter(it,
                                optimizer.param_groups[0]['lr'],
                                result,
                                meta,
                                'train')
            it += 1

        avg_loss = epoch_loss / max(1, len(dataloaders["train"]))
        print(f"[train_salivqa] epoch {epoch:02d} avg_loss {avg_loss:.6f}")

        if save_model:
            save_ckpt(epoch, avg_loss, model)

    return {}
