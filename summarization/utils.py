import random
from transformers import is_torch_available
import torch
import numpy as np
import os
from arguments import train_args
from transformers.trainer_utils import get_last_checkpoint
import logging


# Detecting last checkpoint.
def detect_last_checkpoint(logger):
    last_checkpoint = None
    if os.path.isdir(train_args.args.output_dir) and train_args.args.do_train and not train_args.args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(train_args.args.output_dir)
        if last_checkpoint is None and len(os.listdir(train_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({train_args.args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and train_args.args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    return last_checkpoint

    
# seed 고정하는 함수 (random, numpy, torch)
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print('lock_all_seed')

