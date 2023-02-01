from dataclasses import dataclass
from typing import Dict, Any, Tuple, List
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
from simcls_model import CandidateScorer, CandidateGenerator

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

class SummarizationDataset(torch.utils.data.Dataset):
    def __init__(self, dialogue, summary, candidate, tokenizer, num_docs, num_candidates_per_doc, ID):
        self.tokenizer = tokenizer
        
        self.dialogue = dialogue
        self.summary = summary
        self.candidate = candidate
        self.num_docs = num_docs
        self.num_candidates_per_doc = num_candidates_per_doc

        self.dialogue_token = self.tokenizing(dialogue, 0)
        self.summary_token = self.tokenizing(summary, 0)
        self.candidates_input_ids, self.candidates_attention_mask = self.tokenizing(candidate, 1)
        self.ID = self.data_check(ID)
        self.ori_summary = self.data_check(summary)
    
    def __len__(self):
        return len(self.dialogue)

    def __getitem__(self, idx):
        return {
            "ID" : self.ID[idx],
            "gt_summaries" : self.ori_summary[idx],
            "doc_input_ids" : torch.LongTensor(self.dialogue_token[idx]["input_ids"]).squeeze(0),
            "doc_att_mask" : torch.LongTensor(self.dialogue_token[idx]["attention_mask"]).squeeze(0),
            "candidates_input_ids" : torch.LongTensor(self.candidates_input_ids[idx]).squeeze(0),
            "candidates_att_mask" : torch.LongTensor(self.candidates_attention_mask[idx]).squeeze(0),
            "summary_input_ids" : torch.LongTensor(self.summary_token[idx]["input_ids"]).squeeze(0),
            "summary_att_mask" : torch.LongTensor(self.summary_token[idx]["attention_mask"]).squeeze(0)
            }

    def data_check(self, data):
        data_token = []
        for i in tqdm(data):
            data_token.append(i)
        
        return data_token


    def tokenizing(self, data, num):
        if num == 0:
            data_token = []
            for i in tqdm(data):
                outputs = self.tokenizer(i, padding="max_length", truncation=True, return_tensors="pt", max_length=512)

                data_token.append(outputs)

            return data_token

        else:
            outputs = self.tokenizer(data, padding="max_length", truncation=True, return_tensors="pt", max_length=512)

            candidates_input_ids = outputs["input_ids"].reshape(self.num_docs, self.num_candidates_per_doc, -1)
            candidates_attention_mask = outputs["attention_mask"].reshape(self.num_docs, self.num_candidates_per_doc, -1)

            return candidates_input_ids, candidates_attention_mask

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
    def __init__(self, model: CandidateScorer, generator_path, roberta_path) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.candidate_generator = CandidateGenerator(generator_path, device=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(roberta_path)
        self.model = model.to(self.device)

        self.best_score = -float("inf")
        self.worse_count = 0

    def __batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        inputs = dict()
        for inp in batch:
            if type(batch[inp]) == torch.Tensor:
                inputs[inp] = batch[inp].to(self.device)

        return inputs

    # def collate_inputs_to_batch(self, inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    #     batch = {
    #         "doc_input_ids": torch.tensor([inp["doc_input_ids"] for inp in inputs], dtype=torch.long),
    #         "doc_att_mask": torch.tensor([inp["doc_att_mask"] for inp in inputs], dtype=torch.long),
    #         "candidates_input_ids": torch.tensor([inp["candidates_input_ids"] for inp in inputs], dtype=torch.long),
    #         "candidates_att_mask": torch.tensor([inp["candidates_att_mask"] for inp in inputs], dtype=torch.long),
    #         "summary_input_ids": torch.tensor([inp["summary_input_ids"] for inp in inputs], dtype=torch.long),
    #         "summary_att_mask": torch.tensor([inp["summary_att_mask"] for inp in inputs], dtype=torch.long)
    #     }

    #     return batch

    def train(self, train_dataloader, val_dataloader, config: TrainConfig) -> None:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        scheduler = Scheduler(optimizer, config.lr, config.warmup_steps)
        criterion = RankingLoss(config.margin_lambda)

        # train_dialogue, train_summary = train_dataloader["dialogue"][:300], train_dataloader["summary"][:300]
        # val_dialogue, val_summary = val_dataloader["dialogue"][:300], val_dataloader["summary"][:300]
        
        train_dialogue = [doc for doc in train_dataloader["dialogue"]]
        train_summary  = [doc for doc in train_dataloader["summary"]]
        train_ID       = [doc for doc in train_dataloader["ID"]]
        val_dialogue   = [doc for doc in val_dataloader["dialogue"]]
        val_summary    = [doc for doc in val_dataloader["summary"]]
        val_ID         = [doc for doc in val_dataloader["ID"]]

        print(len(train_dialogue), len(val_dialogue))

        path = "/opt/ml/final-project-level3-nlp-07/summarization/generate_data/"

        train_candidate = [self.candidate_generator([train_dialogue[i]])[0] for i in tqdm(range(len(train_dialogue)))]
        val_candidate = [self.candidate_generator([val_dialogue[i]])[0] for i in tqdm(range(len(val_dialogue)))]

        # with open(path + 'generate_train.pkl', 'wb') as f:            # candidate 저장
        #     pickle.dump(train_candidate, f)

        # with open(path + 'generate_valid.pkl', 'wb') as f:
        #     pickle.dump(val_candidate, f)


        # with open(path + 'generate_train.pkl','rb') as f:             # candidate 불러오기
        #     train_candidate = pickle.load(f)

        # with open(path + 'generate_valid.pkl','rb') as f:
        #     val_candidate = pickle.load(f)

        train_candidate = train_candidate
        val_candidate = val_candidate

        train_candidates = [cand for doc_cands in train_candidate for cand in doc_cands]
        val_candidates = [cand for doc_cands in val_candidate for cand in doc_cands]
        
        dict_candidates = {val_ID[i] : val_candidate[i] for i in tqdm(range(len(val_candidate)))}

        train_num_docs, train_num_candidates_per_doc = len(train_dialogue), len(train_candidate[0])
        val_num_docs, val_num_candidates_per_doc = len(val_dialogue), len(val_candidate[0])

        train_dataset = SummarizationDataset(train_dialogue, train_summary, train_candidates, self.tokenizer, train_num_docs, train_num_candidates_per_doc, train_ID)
        val_dataset = SummarizationDataset(val_dialogue, val_summary, val_candidates, self.tokenizer, val_num_docs, val_num_candidates_per_doc, val_ID)

        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)

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

                optimizer.zero_grad()

                candidate_scores, summary_scores = self.model(**inputs)
                loss = criterion(candidate_scores, summary_scores)
                
                loss.backward()
                scheduler.step()
                optimizer.step()

                epoch_loss_sum += loss.item()
                eval_steps_loss_sum += loss.item()

                if step_counter % config.eval_steps == 0:
                    r1, r2, rl, val_loss = self.evaluate(val_dataloader, criterion, dict_candidates)
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
        r1, r2, rl, val_loss = self.evaluate(val_dataloader, criterion, dict_candidates)
        val_score = (r1 + r2 + rl) / 3
        print(f"[INFO] After {step_counter} steps:"
              f"\n\t- val loss: {val_loss:.4f}, rouge scores: {r1:.4f}/{r2:.4f}/{rl:.4f}")
        self.__handle_early_stopping(val_score, config.early_stopping_patience, config.save_dir)

    def evaluate(self, test_dataloader: DataLoader, criterion: RankingLoss, dict_candidates) -> Tuple[float, float, float, float]:
        self.model.eval()
        rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)

        losses = []
        r1, r2, rl = [], [], []
        with torch.no_grad():
            for batch in tqdm(test_dataloader):

                inputs = self.__batch_to_device(batch)

                candidate_scores, summary_scores = self.model(**inputs)
                loss = criterion(candidate_scores, summary_scores)
                losses.append(loss.item())

                cand_idx = np.argmax(candidate_scores.cpu().numpy(), axis=1)

                candidates = [dict_candidates[cands][idx] for idx, cands in zip(cand_idx, batch["ID"])]

                res = [rouge_scorer.score(summary, cand) for summary, cand in zip(batch["gt_summaries"], candidates)]

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
                self.model.save(save_dir)
            return False
        else:
            self.worse_count += 1
            return self.worse_count >= early_stopping_patience