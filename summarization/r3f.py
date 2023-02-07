from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Union, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from rouge_score.rouge_scorer import RougeScorer
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from metrics import RankingLoss
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizerBase
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast, BartForConditionalGeneration, \
    PegasusTokenizer, PegasusForConditionalGeneration, PreTrainedModel, PreTrainedTokenizer, AutoConfig


import os
import gc
import pickle

torch.cuda.empty_cache()
gc.collect()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class TrainConfig:
    lr: float
    batch_size: int
    num_epochs: int
    save_dir: str = None
    weight_decay: float = 0
    margin_lambda: float = 0.01
    warmup_steps: int = 10000  # as described in the paper
    eval_steps: int = 1000  # in the original work they evaluate every 1000 updates
    early_stopping_patience: int = -1  # -1 don't use early stopping

class DialogueSummarizationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        ids,
        dialogues,
        summaries,
        tokenizer: PreTrainedTokenizerBase,
        dialogue_max_seq_len: int,
        summary_max_seq_len: int,
        use_summary: bool,
    ):
        super().__init__()

        self.sep_token = tokenizer.sep_token
        (
            self.ids,
            self.dialogues,
            self.summaries,
            self.dialogue_input_ids,
            self.dialogue_attention_masks,
            self.summary_input_ids,
            self.summary_attention_masks,
        ) = self.load_dataset(ids, dialogues, summaries, tokenizer, dialogue_max_seq_len, summary_max_seq_len, use_summary)

    def load_dataset(
        self,
        ids,
        dialogues,
        summaries,
        tokenizer: PreTrainedTokenizerBase,
        dialogue_max_seq_len: int,
        summary_max_seq_len: int,
        use_summary: bool,
    ) -> Tuple[
        List[str],
        List[List[str]],
        List[str],
        List[torch.Tensor],
        List[torch.Tensor],
        Optional[List[torch.Tensor]],
        Optional[List[torch.Tensor]],
    ]:

        bos = "<s>"
        eos = "</s>"
        dialogue_inputs = tokenizer(
            [bos + x + eos for x in dialogues],
            padding="max_length",
            truncation=True,
            max_length=dialogue_max_seq_len,
            return_tensors="pt",
            return_token_type_ids=False,
        )

        summary_inputs = (
            tokenizer(
                [bos + x + eos for x in summaries],
                padding="max_length",
                truncation=True,
                max_length=summary_max_seq_len,
                return_tensors="pt",
                return_token_type_ids=False,
            )
            if use_summary
            else {}
        )

        return (
            ids,
            dialogues,
            summaries,
            dialogue_inputs["input_ids"],
            dialogue_inputs["attention_mask"],
            summary_inputs.get("input_ids"),
            summary_inputs.get("attention_mask"),
        )

    def __len__(self) -> int:
        return len(self.dialogue_input_ids)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = {"input_ids": self.dialogue_input_ids[index], "attention_mask": self.dialogue_attention_masks[index]}
        if self.summary_input_ids is not None and self.summary_attention_masks is not None:
            item.update(
                {
                    "decoder_input_ids": self.summary_input_ids[index],
                    "decoder_attention_mask": self.summary_attention_masks[index],
                }
            )
        return item

class Scheduler:
    """
        SimCLS learning rate scheduler as described in paper.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, lr_lambda: float = 0.002, warmup_steps: int = 10000) -> None:
        self.optimizer = optimizer
        self.lr_lambda = lr_lambda
        self.warmup_steps = warmup_steps

        self.step_count = 0

    def step(self) -> None:
        self.step_count += 1
        lr = self.lr_lambda * min(pow(self.step_count, -0.5), self.step_count * pow(self.warmup_steps, -1.5))

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


class Trainer:
    def __init__(self, generator_path) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.config = AutoConfig.from_pretrained(generator_path)
        self.model = BartForConditionalGeneration.from_pretrained(generator_path, config=self.config).to(self.device)
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(generator_path, use_fast=False)
        self.noise_sampler = torch.distributions.normal.Normal(loc=0.0, scale=1e-5)
        self.r3f_lambda = 1.0

    def __batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        inputs = dict()
        for inp in batch:
            if type(batch[inp]) == torch.Tensor:
                inputs[inp] = batch[inp].to(self.device)

        return inputs

    def _get_symm_kl(self, noised_logits, input_logits):
        return (
            F.kl_div(
                F.log_softmax(noised_logits, dim=-1, dtype=torch.float32),
                F.softmax(input_logits, dim=-1, dtype=torch.float32),
                reduction="sum",
            )
            + F.kl_div(
                F.log_softmax(input_logits, dim=-1, dtype=torch.float32),
                F.softmax(noised_logits, dim=-1, dtype=torch.float32),
                reduction="sum",
            )
        ) / noised_logits.size(0)

    def train(self, train_dataloader, val_dataloader, config: TrainConfig) -> None:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        scheduler = Scheduler(optimizer, config.lr, config.warmup_steps)
        criterion = RankingLoss(config.margin_lambda)
                
        train_dialogue = [doc for doc in train_dataloader["dialogue"]]
        train_summary  = [doc for doc in train_dataloader["summary"]]
        train_ID       = [doc for doc in train_dataloader["ID"]]
        val_dialogue   = [doc for doc in val_dataloader["dialogue"]]
        val_summary    = [doc for doc in val_dataloader["summary"]]
        val_ID         = [doc for doc in val_dataloader["ID"]]

        train_dataset = DialogueSummarizationDataset(ids=train_ID, dialogues=train_dialogue, summaries=train_summary,\
        tokenizer=self.tokenizer, dialogue_max_seq_len=256, summary_max_seq_len=64, use_summary=True)

        val_dataset = DialogueSummarizationDataset(ids=val_ID, dialogues=val_dialogue, summaries=val_summary,\
        tokenizer=self.tokenizer, dialogue_max_seq_len=256, summary_max_seq_len=64, use_summary=True)

        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size)

        step_counter = 0
        eval_steps_loss_sum = 0
        self.best_score = -float("inf")
        self.worse_count = 0

        for epoch in range(config.num_epochs):
            epoch_loss_sum = 0

            for i, batch in enumerate(tqdm(train_dataloader)):
                step_counter += 1
                self.model.train()
                inputs = self.__batch_to_device(batch)  # send to GPU

                inputs_embeds = self.model.model.shared(inputs["input_ids"])
                # new_embeddings = super().resize_token_embeddings(new_num_tokens)
                # self.model.shared = new_embeddings


                optimizer.zero_grad()

                output = self.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=inputs["attention_mask"],
                    decoder_input_ids=inputs["decoder_input_ids"],
                    decoder_attention_mask=inputs["decoder_attention_mask"],
                    return_dict=True,
                )

                noise = self.noise_sampler.sample(sample_shape=inputs_embeds.shape).to(inputs_embeds)
                noise_embeds = inputs_embeds.detach().clone() + noise
                noise_output = self.model(
                    inputs_embeds=noise_embeds,
                    attention_mask=inputs["attention_mask"],
                    decoder_input_ids=inputs["decoder_input_ids"],
                    decoder_attention_mask=inputs["decoder_attention_mask"],
                    return_dict=True,
                )

                labels = batch["decoder_input_ids"][:, 1:].reshape(-1).to(self.device)
                logits = output["logits"][:, :-1].reshape([labels.shape[0], -1])
                noise_logits = noise_output["logits"][:, :-1].reshape([labels.shape[0], -1])

                symm_kl = self._get_symm_kl(noise_logits, logits)

                loss = F.cross_entropy(logits, labels, ignore_index=self.model.config.pad_token_id).to(self.device)

                loss += self.r3f_lambda * symm_kl

                loss.backward()
                scheduler.step()
                optimizer.step()
                
                epoch_loss_sum += loss.item()
                eval_steps_loss_sum += loss.item()

                if step_counter % config.eval_steps == 0:
                    r1, r2, rl, val_loss = self.evaluate(val_dataloader, criterion)
                    val_score = (r1 + r2 + rl) / 3
                    train_loss = eval_steps_loss_sum / config.eval_steps
                    eval_steps_loss_sum = 0

                    print(f"[INFO] After {step_counter} steps:\n\t- train loss: {train_loss:.6f}"
                          f"\n\t- val loss: {val_loss:.4f}, rouge scores: {r1:.4f}/{r2:.4f}/{rl:.4f}")

                    should_early_stop = self.__handle_early_stopping(val_score, config.early_stopping_patience,
                                                                     config.save_dir)
                    if should_early_stop:
                        return

            print(f"[INFO] Average loss in epoch {epoch}: {epoch_loss_sum / len(train_dataloader)}")

        # after training is finished, we re-evaluate the model
        r1, r2, rl, val_loss = self.evaluate(val_dataloader, criterion)
        val_score = (r1 + r2 + rl) / 3
        print(f"[INFO] After {step_counter} steps:"
              f"\n\t- val loss: {val_loss:.4f}, rouge scores: {r1:.4f}/{r2:.4f}/{rl:.4f}")
        self.__handle_early_stopping(val_score, config.early_stopping_patience, config.save_dir)

    def evaluate(self, test_dataloader: DataLoader, criterion: RankingLoss) -> Tuple[float, float, float, float]:
        self.model.eval()
        rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)

        losses = []
        r1, r2, rl = [], [], []
        with torch.no_grad():
            for batch in tqdm(test_dataloader):

                inputs = self.__batch_to_device(batch)

                inputs_embeds = self.model.model.shared(inputs["input_ids"])

                val_outputs = self.model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=inputs["attention_mask"],
                    # decoder_input_ids=inputs["decoder_input_ids"],
                    # decoder_attention_mask=inputs["decoder_attention_mask"],
                    # return_dict=True,
                )

                reference_output = "".join(self.tokenizer.batch_decode(inputs["decoder_input_ids"], skip_special_tokens=True))
                val_decode_output = "".join(self.tokenizer.batch_decode(val_outputs, skip_special_tokens=True))

                res = [rouge_scorer.score(reference_output, val_decode_output)]

                r1.extend([r["rouge1"].fmeasure for r in res])
                r2.extend([r["rouge2"].fmeasure for r in res])
                rl.extend([r["rougeLsum"].fmeasure for r in res])

        return np.mean(r1), np.mean(r2), np.mean(rl), np.mean(losses)

    def __handle_early_stopping(self, val_score: float, early_stopping_patience: int, save_dir: str) -> bool:
        if val_score > self.best_score or early_stopping_patience == -1:
            self.best_score = val_score
            self.worse_count = 0

            if hasattr(self.model, "module"):
                self.model.module.save(save_dir)
            else:
                self.model.save_pretrained(save_dir)
            return False
        else:
            self.worse_count += 1
            return self.worse_count >= early_stopping_patience