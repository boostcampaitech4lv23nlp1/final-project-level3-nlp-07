from torch.utils.data import Dataset
import pandas as pd
import random
import re
import torch
from tqdm import tqdm
class trainDataset(Dataset):
    def __init__(self,df, tokenizer) -> None:
        super(trainDataset, self).__init__()
        self.tokenizer = tokenizer
        self.dataset = self._tokenizing(self._preprocessing(df))
        self.label = [0]*len(self.dataset)
    def __len__(self):
        return len(self.dataset)
    def _tokenizing(self,df):
        output = []
        for idx, item in tqdm(df.iterrows(),total=len(df)):
            pos_token = self.tokenizer(item['positive_pair'],add_special_tokens = True, max_length = 128, padding = 'longest',truncation = True,return_tensors = 'pt')
            neg_token_2 = self.tokenizer(item['negative_pair_otehrTopic'],add_special_tokens = True, max_length = 128, padding = 'max_length',truncation = True,return_tensors = 'pt')
            neg_token_1 = self.tokenizer(item['negative_pair2_unadjacent'],add_special_tokens = True, max_length = 128, padding = 'max_length',truncation = True,return_tensors = 'pt')
            output.append([pos_token,neg_token_1])
            output.append([pos_token,neg_token_2])
        return output
    def __getitem__(self, idx):
        # print('neg_input_ids' , self.dataset[idx][1]['input_ids'].squeeze(0))
        return {'input_ids' : self.dataset[idx][0]['input_ids'].squeeze(0),
                 'attention_mask' : self.dataset[idx][0]['attention_mask'].squeeze(0),
                'neg_input_ids' : self.dataset[idx][1]['input_ids'].squeeze(0),
                 'neg_attention_mask' : self.dataset[idx][1]['attention_mask'].squeeze(0),
                 'label' : torch.tensor(self.label[idx])}
    
    def _preprocessing(self,df):
        df['text_boolean'] = df['positive_pair'].apply(self.text_processing)
        df['text_boolean_1'] = df['negative_pair2_unadjacent'].apply(self.text_processing)
        df['text_boolean_2'] = df['negative_pair_otehrTopic'].apply(self.text_processing)
        new_df = df[(df["text_boolean"] == True) & (df["text_boolean_1"] == True) &(df['text_boolean_2'] == True)]
        new_df.reset_index(drop = True, inplace = True)
        return new_df

    def text_processing(self,dialog):                                    # Text 전처리 작업
        find_text = re.findall('[ㄱ-ㅎㅏ-ㅣ]+', dialog)
        vowel = "".join(find_text)
        if vowel == dialog:           # 자음 또는 모음으로만 존재하는 경우
            return False
        if len(dialog) == 0 or len(dialog)<= 5:
            return False
        if dialog == "삭제된 메시지입니다." or dialog == "채팅방 관리자가 메시지를 가렸습니다.":
            return False

        if "님이 나갔습니다." == dialog[-9:] or "님이 들어왔습니다." == dialog[-10:] or "저장한 날짜 : " in dialog:
            return False
        
        if dialog == "이모티콘" or dialog == "사진" or dialog == "카카오톡 프로필" or dialog == "음성메시지" or dialog == "보이스룸이 방금 시작했어요." or \
        dialog[:7] == "보이스룸 종료" or dialog[:7] == "라이브톡 종료" or dialog[:7] == "라이브톡 시작":
            return False
        return True
class DTSDataset(Dataset):
    def __init__(self,df,tokenizer) -> None:
        super(DTSDataset,self).__init__()
        self.tokenizer = tokenizer
        self.label = [0]*len(df)
        self.preprocessed = self._preprocessing(df)
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
        # print('neg_input_ids' , self.dataset[idx][1]['input_ids'].squeeze(0))
        return {'input_ids' : self.dataset[idx]['input_ids'].squeeze(0),
                 'attention_mask' : self.dataset[idx]['attention_mask'].squeeze(0),
                 'label' : torch.tensor(self.label[idx])}
        # return {'input_ids' : [self.dataset[idx][0]['input_ids'].squeeze(0),self.dataset[idx][1]['input_ids'].squeeze(0)],
        #          'attention_mask' : [self.dataset[idx][0]['attention_mask'].squeeze(0),self.dataset[idx][1]['attention_mask'].squeeze(0)],
        #          'label' : torch.tensor(self.label[idx])}
    def _preprocessing(self,df):
        df["id_boolean"] = df["User"].apply(self.id_check)                   # 방장봇이 대화하면 제거
        df["Message"] = df["Message"].apply(self.text_replace)               # \n, 링크 전처리 작업
        df["text_boolean"] = df["Message"].apply(self.text_processing)       # 제거 목록 전처리 작업
        df = df[(df["id_boolean"] == True) & (df["text_boolean"] == True)]     # 전처리 작업 후 (True & True) Data 사용
        df.reset_index(drop=True,inplace = True)
        df['Message2'] = pd.concat([df['Message'].iloc[1:],pd.Series('None')]).reset_index(drop=True)           # window 작업
        df['Date'] = pd.to_datetime(df['Date'],infer_datetime_format=True)                                  # date 날짜화 str -> datetime
        df = df[["Date", "User", "Message","Message2"]]
        # df_filtered2 = df[ df['Date'].isin(pd.date_range('2022-12-16', '2022-12-17',freq = 's'))] # 원하는 일자별로 자를 수도 있음묘 
        return df

    def id_check(self,my_id):                                            # 방장 봇이면 False, 일반 유저인 경우 True
        if my_id == "방장봇":
            return False
        return True
    def text_processing(self,dialog):                                    # Text 전처리 작업
        find_text = re.findall('[ㄱ-ㅎㅏ-ㅣ]+', dialog)
        vowel = "".join(find_text)
        if vowel == dialog:           # 자음 또는 모음으로만 존재하는 경우
            return False
        if len(dialog) == 0 or len(dialog)<= 5:
            return False
        if dialog == "삭제된 메시지입니다." or dialog == "채팅방 관리자가 메시지를 가렸습니다.":
            return False

        if "님이 나갔습니다." == dialog[-9:] or "님이 들어왔습니다." == dialog[-10:] or "저장한 날짜 : " in dialog:
            return False
        
        if dialog == "이모티콘" or dialog == "사진" or dialog == "카카오톡 프로필" or dialog == "음성메시지" or dialog == "보이스룸이 방금 시작했어요." or \
        dialog[:7] == "보이스룸 종료" or dialog[:7] == "라이브톡 종료" or dialog[:7] == "라이브톡 시작":
            return False
        

        return True
    def text_replace(self,dialog):                                       # '\n' -> ' ' , 링크 -> [LINK] 로 변경
        line = "\n"
        dialog = re.sub(pattern=line, repl =" ", string=dialog)

        web = "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        dialog = re.sub(pattern=web, repl ="[LINK]", string=dialog)

        return dialog.strip()