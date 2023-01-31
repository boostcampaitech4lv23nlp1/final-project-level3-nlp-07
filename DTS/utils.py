from tqdm import tqdm
import logging
import torch
import numpy as np
import random
from transformers import Trainer
class TqdmLoggingHandler(logging.StreamHandler):
    """Avoid tqdm progress bar interruption by logger's output to console"""
    # see logging.StreamHandler.eval method:
    # https://github.com/python/cpython/blob/d2e2534751fd675c4d5d3adc208bf4fc984da7bf/Lib/logging/__init__.py#L1082-L1091
    # and tqdm.write method:
    # https://github.com/tqdm/tqdm/blob/f86104a1f30c38e6f80bfd8fb16d5fcde1e7749f/tqdm/std.py#L614-L620

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, end=self.terminator)
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)           # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)                       # 시드를 고정해도 함수를 호출할 때 다른 결과가 나오더라..?
    random.seed(seed)
    print('lock_all_seed')


def compute_metrics(pred):
    acc = 0
    for i in pred:
        if i[0] > i[0]:
            acc +=1
    return {'acc' : acc/len(pred)}

class MarginalTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(inputs)
        logits = outputs['output']        
        loss = self.MarginRankingLoss(logits)
        return (loss, outputs) if return_outputs else loss
    def MarginRankingLoss(self,p_scores, n_scores):
        margin = 1
        scores = margin - p_scores + n_scores
        scores = scores.clamp(min=0)
        return scores.mean()
