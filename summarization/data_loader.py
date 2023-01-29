from datasets import Dataset, DatasetDict, concatenate_datasets
from typing import Dict, List
from model import load_model_tokenizer
from logger import set_logging

import json
import pandas as pd
import torch

logger = set_logging('train')
model, tokenizer = load_model_tokenizer(logger)


# json 형식 파일을 불러오는 함수
def load_json(path: str) -> Dict[str, str]:
    with open(path) as f:
        data = json.load(f)

    return data


# 대화ID, 대화, 요약 정보 추출
def load_data(path: Dict, num) -> List[str]:
    eos = tokenizer.eos_token
    data = load_json(path)
    dialogueID = []
    dialogue = []
    summary = []
    cnt = 0
    for text in data['data']:
        if cnt == 10000 and num == 0:
            break
        if cnt == 1000 and num == 1:
            break
        dialogueID.append(text['header']['dialogueInfo']['dialogueID'])
        summary.append(text['body']['summary'])

        utterances = []
        participantID = ''
        person_dialogue = ''
        for utterance in text['body']['dialogue']:
            if participantID == utterance['participantID']:
                person_dialogue += ' ' + utterance['utterance']
            else:
                if person_dialogue:
                    utterances.append(person_dialogue)
                participantID = utterance['participantID']
                person_dialogue = utterance['utterance']
        if person_dialogue:
            utterances.append(person_dialogue)
        dialogue.append(eos.join(utterances))
        cnt += 1

    data = {}
    data['ID'] = dialogueID
    data['dialogue'] = dialogue
    data['summary'] = summary
    data = Dataset.from_pandas(pd.DataFrame(data))

    return data


# 모든 dataset을 합치고 Dataset 형태로 반환
def load_and_concat_dataset(path: str):
    train_path = path + 'train/'
    valid_path = path + 'valid/'
    # data_list = ['개인및관계', '미용과건강', '상거래(쇼핑)', '시사교육', '식음료', '여가생활', '일과직업', '주거와생활', '행사']
    # data_list = ['시사교육'] # test용
    data_list = ['미용과건강', '상거래(쇼핑)', '시사교육', '식음료', '여가생활', '일과직업', '주거와생활', '행사']
    train_datasets = load_data(train_path + data_list[0] + '.json', 0)
    valid_datasets = load_data(valid_path + data_list[0] + '.json', 1)
    for text in data_list[1:]:
        prev_train_data = load_data(train_path + text + '.json', 0)
        prev_valid_data = load_data(valid_path + text + '.json', 1)
        train_datasets = concatenate_datasets([train_datasets, prev_train_data])
        valid_datasets = concatenate_datasets([valid_datasets, prev_valid_data])
    datasets = DatasetDict({'train' : train_datasets, 'validation' : valid_datasets})

    return datasets