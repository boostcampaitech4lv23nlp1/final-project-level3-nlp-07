from torch import nn
import torch
from collections import OrderedDict

class CSModel(nn.Module):
    def __init__(self) -> None:
        super(CSModel, self).__init__()
        self.model = nn.Sequential(OrderedDict({'Linear' : nn.Linear(768, 768),
                                        'Active_fn' : nn.ReLU(),
                                        'Dropout' : nn.Dropout(p=0.1),
                                        'cls_layer' : nn.Linear(768, 2)}))
    def forward(self,input):
        output = self.model(input)
        return output


class GeneratorParameters:
    num_return_seqs: int = 16
    num_beam_groups: int = 16
    num_beams: int = 16
    no_repeat_ngram_n: int = 3
    diversity_penalty: float = 1.
    length_penalty: float = 2. 

def generate_model(model, tokenizer, input):
    default_parameters = GeneratorParameters()
    inputs = tokenizer(input, padding="longest", truncation=True, return_tensors="pt")
    candidates_input_ids = model.generate(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            num_beams=default_parameters.num_beams,
            length_penalty=default_parameters.length_penalty,
            no_repeat_ngram_size=default_parameters.no_repeat_ngram_n,
            num_return_sequences=default_parameters.num_return_seqs,
            num_beam_groups=default_parameters.num_beam_groups,
            diversity_penalty=default_parameters.diversity_penalty,
        )

    candidates = tokenizer.batch_decode(candidates_input_ids, skip_special_tokens=True)
    return candidates

def tokenizer_input(tokenizer, doc, candidate):
    docs = tokenizer(doc, padding="max_length", truncation=True, return_tensors="pt", max_length=512)
    candidates = tokenizer(candidate, padding="max_length", truncation=True, return_tensors="pt", max_length=512)

    input = {
        "doc_input_ids" : docs["input_ids"],
        "doc_att_mask" : docs["attention_mask"],
        "candidates_input_ids" : candidates["input_ids"],
        "candidates_att_mask" : candidates["attention_mask"]
    }

    return input    

def CandidateScorer(model, doc_input_ids: torch.Tensor, doc_att_mask: torch.Tensor, candidates_input_ids: torch.Tensor,
                candidates_att_mask: torch.Tensor):
    doc_embs = model(input_ids=doc_input_ids, attention_mask=doc_att_mask)[0]
    doc_cls = doc_embs[:, 0, :] 

    num_candidates, seq_len = candidates_input_ids.size()
    candidates_embs = model(input_ids=candidates_input_ids, attention_mask=candidates_att_mask)[0]
    candidates_cls = candidates_embs[:, 0, :].reshape(num_candidates, -1)
    doc_cls = doc_cls.reshape(1, -1).expand(num_candidates, -1)
    return torch.cosine_similarity(doc_cls, candidates_cls, dim=-1)