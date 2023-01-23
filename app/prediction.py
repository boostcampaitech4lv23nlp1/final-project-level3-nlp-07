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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertModel.from_pretrained(config.DTS.BERT_PATH).to(device)
    return model
@st.cache
def load_CSmodel(config) -> CSModel:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CSModel()
    model.load_state_dict(torch.load(config.DTS.cs_PATH)).to(device)
    return model

def get_threshold(scores):
    mu,std = torch.std_mean(scores)
    return mu-(std/2)
def inference_DTS(validation_dataloader,bert_model,cs_model):
    scores = []
    fn = nn.Softmax(dim=1)
    for step, batch in tqdm(enumerate(validation_dataloader),desc = 'inference_step',total = len(validation_dataloader)//32):
        pos_input_ids = batch['input_ids'].to(device)
        pos_input_mask = batch['attention_mask'].to(device)
        with torch.no_grad():
            pos_scores = bert_model(input_ids = pos_input_ids, attention_mask=pos_input_mask).hidden_states[0]
            pos_scores = pos_scores[:,0,:]
            pos_scores = cs_model(pos_scores)
            pos_scores = fn(pos_scores)
        scores += pos_scores[:,0] # batch argmax
    tau = get_threshold(scores)
    label = [ 0 if i >= tau else 1 for i in scores ]
    return label
@st.cache
def get_timeline(df,label):
    timeline = []
    n_topic = sum(label)   # topic의 개수
    seg_idx = 0
    for idx, item in df.iterrows():
        if label[idx] == 1 and (idx - seg_idx) >= 10 : 
            tmp = defaultdict()
            # label == 1 means : segmentation!
            # idx-seg_idx 이전 분절 대비 대화의 최소 개수가 10개는 되야지 스껄
            tmp['start'] = df['Date'].iloc[seg_idx]   # 시작 시점 표시
            tmp['content'] = df['Date'].iloc[seg_idx]   # 시작 시점 표시
            for i in range(seg_idx,idx):
                tmp['dialouge'].append(df['Message'].iloc[i])
            tmp['content'] = (df['Message'].iloc[seg_idx])
            seg_idx = idx
            timeline.append(tmp)
            # keys : id, content, start, summary, dialouge
    return timeline
@st.cache
def get_DTS(bert_model,cs_model,tokenizer,inputs):
    data_collator = DataCollatorWithPadding(tokenizer,padding=True, pad_to_multiple_of=8)
    inference_set = load_dataset.DTSDataset(inputs, tokenizer=tokenizer)
    inference_sampler = SequentialSampler(inference_set)
    inference_dataloader = DataLoader(inference_set, sampler=inference_sampler, batch_size=32,collate_fn=data_collator)
    label = inference_DTS(inference_dataloader,bert_model,cs_model)
    return label
def predict_summary(config, inputs):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = config.summary.API_KEY
    for i in inputs:
        i['summary'] = openai.Completion.create(
                    model="text-davinci-003",
                    prompt='Summarize this for a second-grade student:' + ''.join(i['dialouge']),
                    temperature=0.7,
                    max_tokens=64,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )
    return True