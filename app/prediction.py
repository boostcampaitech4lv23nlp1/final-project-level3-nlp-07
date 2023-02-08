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
import collections
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

# import sys
# sys.path.append("../DTS") # 부모 경로 추가하는 법
import re

from load_dataset import DTSDataset
from transformers import pipeline
from krwordrank.sentence import summarize_with_sentences

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 0
def load_config():
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config
# 0
# def load_DTS(config) -> BertModel:
#     model = BertModel.from_pretrained(config.DTS.BERT_PATH).to(device)
#     return model
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
def inference_DTS(validation_dataloader, cs_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cs_model.to(device)
    scores = []
    for step, batch in tqdm(enumerate(validation_dataloader),desc = 'inference_step',total = len(validation_dataloader)//32):
        pos_input_ids = batch['input_ids'].to(device)
        pos_input_mask = batch['attention_mask'].to(device)
        with torch.no_grad():
            pos_scores = cs_model.inference(input_ids = pos_input_ids, attention_mask=pos_input_mask)        
        scores += pos_scores # batch argmax
    scores = torch.sigmoid(torch.tensor(scores))
    tau = get_threshold(scores)
    scores = scores.detach().cpu().numpy().tolist()
    label = [0 if i >= tau else 1 for i in scores] # 1 주제가 바뀐다. 0 : 안바뀐다
    # label  = [0,0,0,0,0,0,0,1,0,0,0,0,0,1]
    return label

# dialogue에서 keyword 뽑아주는 부분
def keyword_extractor(dialogue):
    # nltk로 명사군의 단어들만 뽑아보기
    to_keywords = [i for i,p in pos_tag(word_tokenize(' '.join(dialogue))) \
                                                    if len(i)>1 and p[:2]=='NN' and i !='..']
    keyword, freq = collections.Counter(to_keywords).most_common(1)[0]
    # TODO : 다른 키워드 처리 방법 찾아보기
    return keyword

def get_timeline(df,label,raw_df,penalty):
    timeline = []
    seg_idx = 0
    for idx in range(len(df)):
        # label == 1 means : segmentation!
        # idx-seg_idx 이전 분절 대비 대화의 최소 개수가 10개는 되도록!
        if label[idx] == 1 and (idx - seg_idx) >= 10 : 
            tmp = {}
            tmp['start'] = str(df.loc[seg_idx,'Date'])   # 시작 시점 표시
            tmp['due'] = str(df.loc[idx,'Date'])
            # 전처리 이전 데이터로 메세지를 모아서 보기 위함
            try:
                tmp['content'] = key_word_extraction(df.loc[seg_idx:idx,'Message'].tolist(),penalty)
            except:
                tmp['content'] = df.loc[seg_idx+1,'Message']
            # tmp['content'] = key_word_extraction(df.loc[seg_idx:idx,'Message'].tolist(),penalty)
            ## raw_df의 index가 string type이어서 str을 씌워주고 .loc을 해야함
            tmp['dialogue'] =df.loc[seg_idx:idx,'Message'].tolist()
            tmp['raw_dialogue'] =raw_df.loc[str(df.loc[seg_idx,'index']):str(df.loc[idx,'index']), 'Message'].tolist()
            seg_idx = idx +1
            timeline.append(tmp)
    return timeline

def get_DTS(cs_model,tokenizer,inputs,penalty):
    # pandas to Dataset
    data_collator = DataCollatorWithPadding(tokenizer,padding=True, pad_to_multiple_of=8)
    inference_set = DTSDataset(inputs, tokenizer=tokenizer)
    inference_processed = inference_set.preprocessed # 전처리 된 데이터셋만 가져온다. 
    inference_sampler = SequentialSampler(inference_set)
    inference_dataloader = DataLoader(inference_set, sampler=inference_sampler, batch_size=32,collate_fn=data_collator)
    label = inference_DTS(inference_dataloader,cs_model)
    timeline = get_timeline(df = inference_processed, label = label, raw_df = inputs,penalty=penalty)
    return timeline

def get_summary_input(input):
    dialogue = input["dialogue"]
    result = '</s>'.join(dialogue)
    return result

def predict_summary(inputs):
    generator = pipeline(model='yeombora/dialogue_summarization')
    output = generator(inputs)[0]['generated_text']
    return f'요약 결과 : {output}'

def key_word_extraction(inputs,penalty):
    PATH = '/opt/ml/input/frontend_Backend_test/final-project-level3-nlp-07/utils/stopword.txt'
    text = [re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s]|[\n]|(@[A-Za-z0-9]+)|(\w+:\/\/\S+)", " ",sent) for sent in inputs]
    with open(PATH,'r') as f:
        words= f.read()
    stopwords = words.split('\n')
    stopwords = {words for words in stopwords}
    # pos = ['채용', '취업', '코테', '인공지능', 'AI','알고리즘','면접','대기업', 'IT기업', 'IT', 'ML','DL','CNN', 'RNN', 'CV', 'NLP','Recsys']
    pnlty = lambda x:0 if any(word in x.split() for word in penalty) else 1
    key_word,sent= summarize_with_sentences(text, min_count=3, max_length=10,stopwords = stopwords,penalty=pnlty,
                                            beta=0.85, max_iter=10, verbose=False)
    return ' #' + ' #'.join(list(key_word.keys())[:3])

def total_key_word_extraction(inputs,penalty):
    PATH = '/opt/ml/input/frontend_Backend_test/final-project-level3-nlp-07/utils/stopword.txt'
    text = [re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s]|[\n]|(@[A-Za-z0-9]+)|(\w+:\/\/\S+)", " ",sent) for sent in inputs]
    with open(PATH,'r') as f:
        words= f.read()
    stopwords = words.split('\n')
    stopwords = {words for words in stopwords}
    # pos = ['채용', '취업', '코테', '인공지능', 'AI','알고리즘','면접','대기업', 'IT기업', 'IT', 'ML','DL','CNN', 'RNN', 'CV', 'NLP','Recsys']
    pnlty = lambda x:0 if any(word in x.split() for word in penalty) else 1
    key_word,sent= summarize_with_sentences(text, min_count=3, max_length=10,stopwords = stopwords,penalty=pnlty,
                                            beta=0.85, max_iter=10, verbose=False)
    return [(k,v) for k, v in key_word.items()][:5]