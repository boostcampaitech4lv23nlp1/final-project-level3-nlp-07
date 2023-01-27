import argparse
import random
from transformers import BertTokenizer
import pandas as pd
import re

def load_df(in_fname):
    df = pd.read_csv(in_fname)
    return df
def prepare_df(df):
    subject_utter =[]
    total_list = list(df['subject'].unique())
    for t in total_list:
        utterance = []
        cat = []
        for i, r in df[(df['subject'] ==t)&(df['speaker_type'] == '다자간 대화')].iterrows():
            utterance += eval(r['norm_texts'])
            cat += eval(r['speech_Act'])
        subject_utter.append([t,utterance, cat])
    new_df = pd.DataFrame(subject_utter, columns=['subject', 'utterance', 'Act'])
    return new_df
def sampling2(item, idx):
    sampled_df = []
        # TODO : positive sample 만들기
    for i in range(len(item['utterance'])-1):
        if item['Act'][i] == '(지시) 질문하기'  and item['Act'][i+1] == '(단언) 진술하기':#
            pos_sample = [item['utterance'][i], item['utterance'][i+1]]
            utterances_wo = [item['utterance'][i] for i in range(len(item['utterance'])-1) if item['Act'][i] != '(단언) 진술하기']
        elif item['Act'][i] == '(주장) 진술하기'  and item['Act'][i+1] == '(지시) 질문하기':#
            pos_sample = [item['utterance'][i], item['utterance'][i+1]]
            utterances_wo = [item['utterance'][i] for i in range(len(item['utterance'])-1) if item['Act'][i] != '(단언) 진술하기']
        elif item['Act'][i] == '(지시) 질문하기'  and item['Act'][i+1] == '(단언) 주장하기':#
            pos_sample = [item['utterance'][i], item['utterance'][i+1]]
            utterances_wo = [item['utterance'][i] for i in range(len(item['utterance'])-1) if item['Act'][i] != '(단언) 진술하기']
        else:
            continue
        ################### negative sample ######################
        # TODO : Other Topics
        neg_topic = random.sample([j for j in range(20) if j != idx], 1)[0]
        neg_sample_1 = [item['utterance'][i],random.sample(new_df.iloc[neg_topic]['utterance'],1)[0]]
        # TODO : same Topics other adjacent
        neg_smp = random.sample(utterances_wo, 1)[0]
        neg_sample_2 = [item['utterance'][i],neg_smp]
        sampled_df.append([pos_sample, neg_sample_1, neg_sample_2])
    print(f'topic {idx} is done')
    sampled_df = pd.DataFrame(sampled_df)
    return sampled_df

def sample(new_df,PATH):
    tmp = pd.DataFrame()
    for idx, item in new_df.iterrows():
        tmp = pd.concat([tmp,sampling2(item,idx)],axis = 0)
    tmp.columns = ['positive_pair', 'negative_pair_otehrTopic', 'negative_pair2_unadjacent']
    tmp.to_csv(PATH,index=False)

if __name__ == '__main__':
    v_PATH = '/opt/ml/input/data/dialogue/valid.csv'
    path = '/opt/ml/input/poc/valid (1).csv'
    print('now valid')
    df = prepare_df(load_df(path))
    sample(df,v_PATH)
