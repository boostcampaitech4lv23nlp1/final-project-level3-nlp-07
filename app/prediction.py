from model import CSModel
from torch import nn
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import Automodel, BertModel, DataCollatorWithPadding
from collections import defaultdict
import yaml
import streamlit as st
import os
from tqdm.auto import tqdm
import openai
from DTS import load_dataset
from DTS import model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
@st.cache
def load_config():
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config
@st.cache
def load_DTS(config) -> BertModel:
    model = BertModel.from_pretrained(config.DTS.BERT_PATH).to(device)
    return model
@st.cache
def load_CSmodel(config) -> CSModel:
    model = CSModel()
    model.load_state_dict(torch.load(config.DTS.cs_PATH))
    model.to(device)
    return model

def get_threshold(scores):
    std,mu = torch.std_mean(scores)
    return mu-(std/1.5)
def inference_DTS(validation_dataloader,bert_model,cs_model):
    scores = []
    for step, batch in tqdm(enumerate(validation_dataloader),desc = 'inference_step',total = len(validation_dataloader)//32):
        pos_input_ids = batch['input_ids'].to(device)
        pos_input_mask = batch['attention_mask'].to(device)
        with torch.no_grad():
            pos_scores = bert_model(input_ids = pos_input_ids, attention_mask=pos_input_mask).hidden_states[0]
            pos_scores = pos_scores[:,0,:]
            pos_scores = cs_model(pos_scores)
        
        scores += pos_scores[:,0] # batch argmax
    scores = torch.sigmoid(torch.tensor(scores))
    tau = get_threshold(scores)
    scores = scores.detach().cpu().numpy().tolist()
    label = [ 0 if i >= tau else 1 for i in scores]
    return label
@st.cache
def get_timeline(df,label,raw_df):
    timeline = []
    seg_idx = 0
    for idx in range(len(df)):
        if label[idx] == 1 and (idx - seg_idx) >= 10 : 
            tmp = {}
            # label == 1 means : segmentation!
            # idx-seg_idx 이전 분절 대비 대화의 최소 개수가 10개는 되야지
            tmp['start'] = str(df['Date'].iloc[seg_idx])   # 시작 시점 표시
            tmp['content'] = (df['Message'].iloc[seg_idx])
            st_point = raw_df[raw_df['Date'] == str(df['Date'].iloc[seg_idx])].index.tolist()[0]
            end_point = raw_df[raw_df['Date'] == str(df['Date'].iloc[idx])].index.tolist()[0]
            tmp['dialouge'] =raw_df['Message'].iloc[st_point:end_point+1].tolist() # end_point까지 모집
            seg_idx = idx +1
            timeline.append(tmp)
            # keys : id, content, start, summary, dialouge
    return timeline

@st.cache
def get_DTS(bert_model,cs_model,tokenizer,inputs):
    data_collator = DataCollatorWithPadding(tokenizer,padding=True, pad_to_multiple_of=8)
    inference_set = load_dataset.DTSDataset(inputs, tokenizer=tokenizer)
    inference_processed = inference_set.preprocessed # 전처리 된 데이터셋만 가져온다. 
    inference_sampler = SequentialSampler(inference_set)
    inference_dataloader = DataLoader(inference_set, sampler=inference_sampler, batch_size=32,collate_fn=data_collator)
    label = inference_DTS(inference_dataloader,bert_model,cs_model)
    return inference_processed,label

def predict_summary(config, inputs):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = config.summary.API_KEY
    for i in inputs:
        #i['summary'] = openai.Completion.create(
        response = openai.Completion.create(
                    model="text-davinci-003",
                    prompt='Summarize this for a second-grade student:' + ''.join(i['dialouge']),
                    temperature=0.7,
                    max_tokens=64,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )
        i['summary'] = response['choices'][0]['text']
    return True