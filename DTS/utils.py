from tqdm import tqdm
import logging
import torch
import numpy as np
import random
from transformers import Trainer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler,Dataset
from transformers.utils import is_datasets_available
from transformers.trainer_pt_utils import IterableDatasetShard
from typing import Optional
from transformers.trainer_utils import EvalPrediction
import datasets
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
    output = pred.predictions['output']
    ls = (output['pos'] > output['neg']).squeeze(0).detach().cpu().numpy().tolist()
    acc =0
    for i in ls:
        acc += int(i[0])
    return {'acc' : acc/len(ls)}

class MarginalTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(inputs)
        logits = outputs['output']    

        loss_fn  = self.MarginRankingLoss
        loss = loss_fn(logits['pos'], logits['neg'])
        # loss = self.MarginRankingLoss(logits['pos'], logits['neg'])
        return (loss, outputs) if return_outputs else loss
    def MarginRankingLoss(self,p_scores, n_scores):
        margin = 1
        scores = margin - p_scores + n_scores
        scores = scores.clamp(min=0)
        return scores.mean()
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].
        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.
        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        train_sampler = RandomSampler(train_dataset)

        return DataLoader(
            train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator
        )
    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].
        Subclass and override this method if you want to inject some custom behavior.
        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.data_collator
        eval_sampler = SequentialSampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=data_collator
        )