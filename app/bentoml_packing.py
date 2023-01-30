from model import CSModel
import bentoml
from bentoml import env, artifacts, api
from bentoml.adapters import JsonInput
from bentoml.frameworks.transformers import TransformersModelArtifact
from bentoml.service.artifacts.common import PickleArtifact
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast, pipeline, AutoTokenizer, BertModel
from prediction import *
import torch


@env(infer_pip_packages=True)
@artifacts([
    TransformersModelArtifact('kobart_model'),
    TransformersModelArtifact('bert_10'),
    PickleArtifact('cs_10')])
class SummaryService(bentoml.BentoService):
    @api(input=JsonInput(), batch=False)
    def dts(self, input):
        dts_tokenizer = self.artifacts.bert_10.get("tokenizer")
        bert_model = self.artifacts.bert_10.get("model")
        cs_model = self.artifacts.cs_10.get("model")
        timeline = get_DTS(bert_model, cs_model, dts_tokenizer, input)
        return timeline


    @api(input=JsonInput(), batch=False)
    def summarization(self, parsed_json):
        dialogue = parsed_json.get("dialogue")
        result = ""
        for i in range(len(dialogue)):
            if i == 0:
                result += dialogue[i]
            else:
                result += '</s>' + dialogue[i]
        model = self.artifacts.kobart_model.get("model")
        tokenizer = self.artifacts.kobart_model.get("tokenizer")
        model_name = "yeombora/dialogue_summarization"
        generator = pipeline(model=model_name)
        output = generator(result)[0]['generated_text']
        return output


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dts_tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
    dts_bert_model = BertModel.from_pretrained('/opt/ml/input/poc/BERT/bert_10').to(device)  

    dts_cs_model = CSModel()  
    dts_cs_model.load_state_dict(torch.load('/opt/ml/input/poc/BERT/cs10.pt'))
    dts_cs_model.to(device)  

    summary_model = BartForConditionalGeneration.from_pretrained("yeombora/dialogue_summarization").to(device)  
    summary_tokenizer = PreTrainedTokenizerFast.from_pretrained("yeombora/dialogue_summarization", use_fast=True)

    bento_svc = SummaryService()

    artifact = {"model": summary_model, "tokenizer": summary_tokenizer}
    bento_svc.pack("kobart_model", artifact)

    artifact = {"model" : dts_bert_model, "tokenizer" : dts_tokenizer}
    bento_svc.pack("bert_10", artifact)

    artifact = {"model" : dts_cs_model}
    bento_svc.pack("cs_10", artifact)
    
    saved_path = bento_svc.save()