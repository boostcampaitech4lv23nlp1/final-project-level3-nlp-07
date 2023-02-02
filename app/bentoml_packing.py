from model import CSModel, generate_model, tokenizer_input, CandidateScorer
import bentoml
from bentoml import env, artifacts, api
from bentoml.adapters import JsonInput
from bentoml.frameworks.transformers import TransformersModelArtifact
from bentoml.service.artifacts.common import PickleArtifact
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast, pipeline, AutoTokenizer, BertModel, AutoModel
from prediction import *
import torch
import numpy as np


@env(infer_pip_packages=True)
@artifacts([
    TransformersModelArtifact('kobart_model'),
    PickleArtifact('cs_10'),
    TransformersModelArtifact('simcls')])
class SummaryService(bentoml.BentoService):
    @api(input=JsonInput(), batch=False)
    def dts(self, inputs,penalty):
        dts_tokenizer = self.artifacts.cs_10.get("tokenizer")
        cs_model = self.artifacts.cs_10.get("model")
        timeline = get_DTS(cs_model, dts_tokenizer, inputs,penalty)
        return timeline


    @api(input=JsonInput(), batch=False)
    def summarization(self, parsed_json):
        bart_model = self.artifacts.kobart_model.get("model")
        bart_tokenizer = self.artifacts.kobart_model.get("tokenizer")
        roberta_model = self.artifacts.simcls.get("model")
        roberta_tokenizer = self.artifacts.simcls.get("tokenizer")

        dialogue = parsed_json.get("dialogue")
        input = '</s>'.join(dialogue)
        
        candidates = generate_model(bart_model, bart_tokenizer, input)
        inputs = tokenizer_input(roberta_tokenizer, input, candidates)
        score = CandidateScorer(roberta_model, **inputs)
        score = score.detach().numpy()
        indices = np.argmax(score, axis=-1)

        return candidates[indices]


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dts_tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
    # dts_bert_model = BertModel.from_pretrained('/opt/ml/input/poc/BERT/bert_10').to(device)  

    dts_cs_model = CSModel()  
    dts_cs_model.load_state_dict(torch.load('/opt/ml/input/poc/Domain_Topic_Sgmentation/Dialog_Topic_segmentation_v1/Domain_Topic_segmentor.pt'))
    dts_cs_model.to(device)  

    summary_tokenizer = PreTrainedTokenizerFast.from_pretrained("yeombora/dialogue_summarization", use_fast=True)
    summary_model = BartForConditionalGeneration.from_pretrained("yeombora/dialogue_summarization").to(device)  

    roberta_tokenizer = AutoTokenizer.from_pretrained("yeombora/SimCLS_Test")  
    roberta_model = AutoModel.from_pretrained("yeombora/SimCLS_Test").to(device)  

    bento_svc = SummaryService()

    artifact = {"model": summary_model, "tokenizer": summary_tokenizer}
    bento_svc.pack("kobart_model", artifact)

    # artifact = {"model" : None , "tokenizer" : dts_tokenizer}
    # bento_svc.pack("bert_10", artifact)

    artifact = {"model" : dts_cs_model,  "tokenizer" : dts_tokenizer}
    bento_svc.pack("cs_10", artifact)

    artifact = {"model" : roberta_model, "tokenizer" : roberta_tokenizer}
    bento_svc.pack("simcls", artifact)
    
    saved_path = bento_svc.save()