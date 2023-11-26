import os
import torch
import torch.distributed as dist

from tqdm import tqdm
from models.memory import MemoryTrace
from models.distillation_model import (preprocess_distillation_batch, distil_loss)

def evaluation(model, train_config, distil_config, eval_dataloader, steps_per_eval, local_rank):
    if train_config.enable_fsdp or distil_config.enable_fsdp: world_size = int(os.environ["WORLD_SIZE"])
    model.eval()
    eval_loss = 0.0
    eval_cross_loss = 0.0
    eval_dist_loss = 0.0
    pbar = tqdm(colour="green", desc="Evaluating", total=steps_per_eval, dynamic_ncols=True)
    for step, batch in enumerate(eval_dataloader):
        if train_config.distillation:
            batch = preprocess_distillation_batch(batch)
        for key in batch.keys():
            if train_config.enable_fsdp or distil_config.enable_fsdp:
                batch[key] = batch[key].to(local_rank)
            else:
                batch[key] = batch[key].to('cuda:0')

        with torch.no_grad():
            if train_config.distillation:
                outputs, teacher_output = model(**batch)
                loss, cross_loss, dist_loss = distil_loss(outputs, teacher_output, batch['student_labels'], batch['teacher_labels'])
                eval_cross_loss += cross_loss.detach().float()
                eval_dist_loss += dist_loss.detach().float()
            else:
                outputs = model(**batch)
                loss = outputs.loss
            eval_loss += loss.detach().float()

        pbar.set_description(f"Evaluationg step {step}/{steps_per_eval}")

    # If there's more than one CUDA device, reduce evaluation loss across all devices
    if torch.cuda.device_count() > 1 and train_config.enable_fsdp or distil_config.enable_fsdp:
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)

    # Compute average loss and perplexity
    eval_epoch_loss = eval_loss / steps_per_eval
    if train_config.enable_fsdp or distil_config.enable_fsdp:
        eval_epoch_loss = eval_epoch_loss/world_size
    eval_ppl = torch.exp(eval_epoch_loss)

    return eval_ppl, eval_epoch_loss, eval_cross_loss, eval_dist_loss