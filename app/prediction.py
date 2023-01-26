from model import CSModel
from torch import nn
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertModel, DataCollatorWithPadding
from collections import defaultdict
import yaml
import streamlit as st
import os
from tqdm.auto import tqdm
import openai
import collections
import sys
sys.path.append("../DTS") # 부모 경로 추가하는 법
from load_dataset import DTSDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 0
def load_config():
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config
# 0
def load_DTS(config) -> BertModel:
    model = BertModel.from_pretrained(config.DTS.BERT_PATH).to(device)
    return model
# 0
def load_CSmodel(config) -> CSModel:
    model = CSModel()
    model.load_state_dict(torch.load(config.DTS.cs_PATH))
    model.to(device)
    return model
# 2
def get_threshold(scores):
    std,mu = torch.std_mean(scores)
    return mu-(std/1.5)
# 1
# 배치 단위 서빙을해야한다.
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
    label = [0 if i >= tau else 1 for i in scores] # 1 주제가 바뀐다. 0 : 안바뀐다
    # label  = [0,0,0,0,0,0,0,1,0,0,0,0,0,1]
    return label

# dialogue에서 keyword 뽑아주는 부분
def keyword_extractor(dialogue):
    # 두음절 이상만 뽑아보기
    to_keywords = [i for i in ' '.join(dialogue).split() if len(i)>1]
    # TODO : keyword 정제하기 (형태소 기반, 빈도수 기반)
    # nltk 설치 완료 후에 진행할 부분
    keyword, freq = collections.Counter(to_keywords).most_common(1)[0]
    return keyword

def get_timeline(df,label,raw_df):
    timeline = []
    seg_idx = 0
    for idx in range(len(df)):
        # label == 1 means : segmentation!
        # idx-seg_idx 이전 분절 대비 대화의 최소 개수가 10개는 되도록!
        if label[idx] == 1 and (idx - seg_idx) >= 10 : 
            tmp = {}
            tmp['start'] = str(df.loc[seg_idx,'Date'])   # 시작 시점 표시
            tmp['content'] = keyword_extractor(df.loc[seg_idx:idx,'Message'])
            # 전처리 이전 데이터로 메세지를 모아서 보기 위함
            # st_point, end_point는 raw_df 기준
            st_point = raw_df[raw_df['Date'] == str(df.loc[seg_idx,'Date'])].index.tolist()[0]
            end_point = raw_df[raw_df['Date'] == str(df.loc[idx,'Date'])].index.tolist()[0]
            tmp['dialogue'] =raw_df.loc[st_point:end_point+1,'Message'].tolist() # end_point까지 모집
            seg_idx = idx +1
            timeline.append(tmp)
    return timeline

def get_DTS(bert_model,cs_model,tokenizer,inputs):
    # pandas to Dataset
    data_collator = DataCollatorWithPadding(tokenizer,padding=True, pad_to_multiple_of=8)
    inference_set = DTSDataset(inputs, tokenizer=tokenizer)
    inference_processed = inference_set.preprocessed # 전처리 된 데이터셋만 가져온다. 
    inference_sampler = SequentialSampler(inference_set)
    inference_dataloader = DataLoader(inference_set, sampler=inference_sampler, batch_size=32,collate_fn=data_collator)
    label = inference_DTS(inference_dataloader,bert_model,cs_model)
    timeline = get_timeline(df = inference_processed, label = label, raw_df = inputs)
    return timeline

def predict_summary(inputs):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    # openai.api_key = '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'
    # user = inputs['USER_ID']
    # user = inputs['dialouge']
    # response = openai.Completion.create(
    #                 model="text-davinci-003",
    #                 prompt='Summarize this for a second-grade student:' + ''.join(inputs['dialouge']),
    #                 frequency_penalty=0.0,
    #                 presence_penalty=0.0
    #             )
    # 주제가 분절된 것을 기준으로 하나의 주제만 summary 한다고 보시면 됩니다!!
    return f'test 입니다. {inputs["dialogue"][-9]}'