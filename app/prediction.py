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
    label = [ 0 if i >= tau else 1 for i in scores] # 1 주제가 바뀐다. 0 : 안바뀐다
    # label  = [0,0,0,0,0,0,0,1,0,0,0,0,0,1]
    return label
# 2
def get_timeline(df,label,raw_df):
    timeline = []
    seg_idx = 0
    # keys : id, content, start, summary, dialouge
    # label  = [0,0,0,0,0,0,0,1,0,0,0,0,0,1]
    # for 1 ; seg_idx = 11 , idx = 23 label[idx] = 1
    for idx in range(len(df)): # len(df) == len(label) , len(label) != len(raw_df)
        if label[idx] == 1 and (idx - seg_idx) >= 10 : # (idx - seg_idx) >= 10 대화가 적어도 10번은 오고 가야지 짜샤
            tmp = {}
            # label == 1 means : segmentation!
            # idx-seg_idx 이전 분절 대비 대화의 최소 개수가 10개는 되야지
            tmp['start'] = str(df['Date'].iloc[seg_idx])   # 시작 시점 표시
            tmp['content'] = (df['Message'].iloc[seg_idx]) # 키워드 발전 여지 있는 부분
            # 전처리 이전 데이터로 메세지를 모아서 보기 위함
            # st_point, end_point는 raw_df 기준
            st_point = raw_df[raw_df['Date'] == str(df['Date'].iloc[seg_idx])].index.tolist()[0]
            end_point = raw_df[raw_df['Date'] == str(df['Date'].iloc[idx])].index.tolist()[0]
            tmp['dialouge'] =raw_df['Message'].iloc[st_point:end_point+1].tolist() # end_point까지 모집
            # tmp['USER_ID'] =raw_df['ID'].iloc[st_point:end_point+1].tolist() # end_point까지 모집
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
    timeline = get_timeline(df = inference_processed,label = label,raw_df = inputs)
    return timeline
# 3
def predict_summary(inputs):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = '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'
    # user = inputs['USER_ID']
    # user = inputs['dialouge']
    # response = openai.Completion.create(
    #                 model="text-davinci-003",
    #                 prompt='Summarize this for a second-grade student:' + ''.join(inputs['dialouge']),
    #                 temperature=0.7,
    #                 max_tokens=128,
    #                 top_p=1.0,
    #                 frequency_penalty=0.0,
    #                 presence_penalty=0.0
    #             )
    # 주제가 분절된 것을 기준으로 하나의 주제만 summary 한다고 보시면 됩니다!!
    return f'test 입니다. {inputs["dialouge"][-9]}'