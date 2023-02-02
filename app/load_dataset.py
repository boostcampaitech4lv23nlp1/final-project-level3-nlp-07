from torch.utils.data import Dataset
import pandas as pd
import random
import re
import torch
from tqdm import tqdm
import sys
sys.path.append("../utils") # 부모 경로 추가하는 법
from preprocessing import _preprocess

class DTSDataset(Dataset):
    def __init__(self,df,tokenizer) -> None:
        super(DTSDataset,self).__init__()
        self.tokenizer = tokenizer
        self.label = [0]*len(df)
        self.preprocessed = _preprocess(df)
        self.dataset = self._tokenizing(self.preprocessed)
    
    def _tokenizing(self,df):
        output =[]
        for idx, item in tqdm(df.iterrows(),total=len(df)):
            neg_idx = random.sample([i for i in range(len(df)) if i != idx], 1)[0]#len(idx))
            # print(neg_idx,idx)
            # print('-'*100, type(self.data.iloc[neg_idx]['Message']), self.data.iloc[neg_idx]['Message'])
            pos_token = self.tokenizer(item['Message'],item['Message2'],add_special_tokens = True, max_length = 128, padding = 'longest',truncation = True,return_tensors = 'pt')
            output.append(pos_token)
        return output

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {'input_ids' : self.dataset[idx]['input_ids'].squeeze(0),
                 'attention_mask' : self.dataset[idx]['attention_mask'].squeeze(0),
                 'label' : torch.tensor(self.label[idx])}
