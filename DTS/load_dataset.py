from torch.utils.data import Dataset
import pandas as pd
import random
import re
import torch
from tqdm import tqdm
import sys
sys.path.append("../utils") # 부모 경로 추가하는 법
from preprocessing import _preprocess


def load_data(PATH):
    df =pd.read_csv(PATH)
    return df
class TrainDataset(Dataset):
    def __init__(self,df, tokenizer) -> None:
        super(TrainDataset, self).__init__()
        self.tokenizer = tokenizer
        self.dataset = self._tokenizing(df)
        self.label = [[0,1] for _ in range(len(self.dataset))]
    def __len__(self):
        return len(self.label)
    def _tokenizing(self,df):
        output = []
        for idx, item in tqdm(df.iterrows(),total=len(df)):
            pos_token = self.tokenizer('[SEP]'.join(eval(item['positive_pair'])),add_special_tokens = True, max_length = 128, padding = True,truncation = True,return_tensors = 'pt')
            neg_token_2 = self.tokenizer('[SEP]'.join(eval(item['negative_pair_otehrTopic'])),add_special_tokens = True, max_length = 64, padding = 'max_length',truncation = True,return_tensors = 'pt')
            neg_token_1 = self.tokenizer('[SEP]'.join(eval(item['negative_pair2_unadjacent'])),add_special_tokens = True, max_length = 64, padding = 'max_length',truncation = True,return_tensors = 'pt')
            output.append([pos_token,neg_token_1])
            output.append([pos_token,neg_token_2])
        return output
    def __getitem__(self, idx):
        # print(self.dataset[idx])
        return {'input_ids' :torch.LongTensor(self.dataset[idx][0]['input_ids']).squeeze(0),
                 'attention_mask' : torch.LongTensor(self.dataset[idx][0]['attention_mask']).squeeze(0),
                'neg_input_ids' : torch.LongTensor(self.dataset[idx][1]['input_ids']).squeeze(0),
                 'neg_attention_mask' : torch.LongTensor(self.dataset[idx][1]['attention_mask']).squeeze(0),
                 'labels' : torch.tensor(self.label[idx]).squeeze(0)}