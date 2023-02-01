from dataclasses import dataclass
from enum import Enum
from typing import Union, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import gc
import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast, BartForConditionalGeneration, \
    PegasusTokenizer, PegasusForConditionalGeneration, PreTrainedModel, PreTrainedTokenizer, AutoConfig

gc.collect()
torch.cuda.empty_cache()

@dataclass
class GeneratorParameters:
    num_return_seqs: int = 16
    num_beam_groups: int = 16
    num_beams: int = 16
    no_repeat_ngram_n: int = 3
    diversity_penalty: float = 1.
    length_penalty: float = 2.

class CandidateGenerator:
    
    default_parameters = GeneratorParameters()

    def __init__(self, path: str, device: torch.device = None, **kwargs) -> None:
        super(CandidateGenerator, self).__init__()

        self.parameters = GeneratorParameters()
        # self.parameters = __get_params(cfg)

        self.device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.generator, self.tokenizer = self.__get_generator_and_tokenizer(path)

        self.generator = self.generator.eval().to(self.device)

    @torch.no_grad()
    def forward(self, docs: List[str]) -> List[List[str]]:

        inputs = self.tokenizer(docs, padding="longest", truncation=True, return_tensors="pt")

        candidates_input_ids = self.generator.generate(
            input_ids=inputs["input_ids"].to(self.device), attention_mask=inputs["attention_mask"].to(self.device),
            early_stopping=True,
            num_beams=self.parameters.num_beams,
            length_penalty=self.parameters.length_penalty,
            no_repeat_ngram_size=self.parameters.no_repeat_ngram_n,
            num_return_sequences=self.parameters.num_return_seqs,
            num_beam_groups=self.parameters.num_beam_groups,
            diversity_penalty=self.parameters.diversity_penalty,
        )

        batched_cands = self.tokenizer.batch_decode(candidates_input_ids, skip_special_tokens=True)
        cands_per_doc = len(batched_cands) // len(docs)
        candidates = [batched_cands[(i * cands_per_doc):((i + 1) * cands_per_doc)] for i in range(len(docs))]

        return candidates

    def __call__(self, docs: List[str], **kwargs) -> List[List[str]]:
        return self.forward(docs, **kwargs)

    def __get_params(self, path: str, num_return_seqs: int = None, num_beam_groups: int = None, num_beams: int = None,
                     no_repeat_ngram_n: int = None, diversity_penalty: float = None, length_penalty: float = None) \
            -> GeneratorParameters:

        parameters = GeneratorParameters(
            num_return_seqs=num_return_seqs if num_return_seqs is not None else default_parameters.num_return_seqs,
            num_beam_groups=num_beam_groups if num_beam_groups is not None else default_parameters.num_beam_groups,
            num_beams=num_beams if num_beams is not None else default_parameters.num_beams,
            no_repeat_ngram_n=no_repeat_ngram_n if no_repeat_ngram_n is not None else default_parameters.no_repeat_ngram_n,
            diversity_penalty=diversity_penalty if diversity_penalty is not None else default_parameters.diversity_penalty,
            length_penalty=length_penalty if length_penalty is not None else default_parameters.length_penalty,
        )

        return parameters

    def __get_generator_and_tokenizer(self, path: str = None) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:

        return BartForConditionalGeneration.from_pretrained(path), PreTrainedTokenizerFast.from_pretrained(path)

class CandidateScorer(nn.Module):

    def __init__(self, path: str) -> None:
        super(CandidateScorer, self).__init__()

        self.encoder = AutoModel.from_pretrained(path)

    def forward(self, doc_input_ids: torch.Tensor, doc_att_mask: torch.Tensor, candidates_input_ids: torch.Tensor,
                candidates_att_mask: torch.Tensor, summary_input_ids: torch.Tensor = None,
                summary_att_mask: torch.Tensor = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        # calculates document representation
        doc_embs = self.encoder(input_ids=doc_input_ids, attention_mask=doc_att_mask)[0]
        doc_cls = doc_embs[:, 0, :]  # CLS is the first token in the sequence

        # calculates representations of all candidate summaries
        # encoder expects a two-dimensional input tensor
        batch_size, num_candidates, seq_len = candidates_input_ids.size()

        candidates_input_ids = candidates_input_ids.reshape(batch_size * num_candidates, seq_len)
        candidates_att_mask = candidates_att_mask.reshape(batch_size * num_candidates, seq_len)
        candidates_embs = self.encoder(input_ids=candidates_input_ids, attention_mask=candidates_att_mask)[0]
        candidates_cls = candidates_embs[:, 0, :].reshape(batch_size, num_candidates, -1)

        if summary_input_ids is None:
            doc_cls = doc_cls.reshape(batch_size, 1, -1).expand(batch_size, num_candidates, -1)
            return torch.cosine_similarity(doc_cls, candidates_cls, dim=-1)

        # calculates reference summary representation
        summary_embs = self.encoder(input_ids=summary_input_ids, attention_mask=summary_att_mask)[0]
        summary_cls = summary_embs[:, 0, :]

        ref_summary_scores = torch.cosine_similarity(doc_cls, summary_cls, dim=-1)
        doc_cls = doc_cls.reshape(batch_size, 1, -1).expand(batch_size, num_candidates, -1)
        candidates_scores = torch.cosine_similarity(doc_cls, candidates_cls, dim=-1)

        return candidates_scores, ref_summary_scores

    def save(self, path: str) -> None:
        self.encoder.save_pretrained(path)


