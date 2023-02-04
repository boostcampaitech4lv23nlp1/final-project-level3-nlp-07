from fastapi import FastAPI
from pydantic import BaseModel, Field
from pymongo import MongoClient
from uuid import UUID, uuid4
from typing import List, Union, Optional, Dict, Any
import pandas as pd 
import json
from datetime import datetime, timedelta
import time
import requests

## Todo: Backend

## Todo: FastAPI Server 띄우기

app = FastAPI()

## Todo: /login
    ## Todo: Frontend에서 login 정보가 넘어옴
    ## Todo: login 정보대로 DB에 접근, 카톡방 목록 가져옴
    ## Todo: 카톡방 목록을 frontend에 전달
    
class User(BaseModel):
    id : Optional[str] = None
    password : Optional[str] = None

class Chatlist(BaseModel):
    chat_room : str

class Chat(BaseModel):
    Date: str
    User: str
    Message: str

@app.post('/login', description="user 정보를 가져옵니다.")
def match_password(item : User):
    root_password = user_db['users'].find_one({'user_id':item.id})['password']
    if root_password == item.password:
        return True
    return False

@app.post('/chatlist', description='채팅방 정보를 가져옵니다.')
def get_chatlist(item : User):
    chat = []
    for d in user_db['user_chat_join'].find({'user_id':item.id}):
        chat.append(d["chat_id"])
    return chat

# 기간 설정이 여기 들어갈 수 있습니다.
# @app.post('/messages', description='채팅을 가져옵니다.')
def get_chattings(item):
    chat = []
    for d in chat_db[item['chat_room']].find():
        chat.append({"Date":d['Date'],"User":d['User'],"Message":d['Message']})
    return chat

def get_now(start_date, time_period, df):    
    df['Date'] = pd.to_datetime(df['Date'],infer_datetime_format=True)
    sample = df[df['Date'].isin(pd.date_range(str(start_date), str(start_date + timedelta(days=time_period)),freq = 's'))]
    return sample

## Todo: /dts
    ## Todo: Frontend에서 카톡방, 날짜 선택해서 그에 맞는 정보가 넘어옴
    ## Todo: 그에 맞는 정보대로 DB에 접근
    ## Todo: sample 만들어서 BentoML server와 통신하여 DTS output 받아오기
    ## Todo: DTS output을 Frontend로 전달

class DtsInput(BaseModel):
    chat_room : str
    start_date : str ## or datetime, 아마 string일 듯
    time_period : str
    penalty : List[str]

@app.post('/dts')
def make_dts(item : DtsInput):

    chatroom = item.chat_room
    chat_dict = {"chat_room": chatroom}
    
    ## chat_dict를 활용하여 DB에 접근해 chattings를 가져옴
    chat = get_chattings(chat_dict)

    ## 가져온 chattings를 DataFrame으로 변환하고 index 만들어줌 (그냥은 index가 없는데 추후 get_DTS에 필요함)
    chat_df = pd.DataFrame(chat).reset_index()
    
    ## str으로 넘어온 start_date를 datetime class로 바꿔줌
    temp = item.start_date
    timestamp = datetime.strptime(temp, '%Y-%m-%d') ## 임시 : str to datetime

    ## get_now를 통과하여 start_date ~ start_date + time_period 까지 sampling 진행
    sample = get_now(timestamp, int(item.time_period), chat_df)

    ## JSON 형태로 넘기기 위해 datetime -> str으로 바꿔주고 / DataFrame을 dict로 변환
    sample['Date'] = sample['Date'].apply(str)
    item = sample.to_dict()

    ## bentoml url for api dts
    url = 'http://0.0.0.0:5000/dts'

    ## 추후 penalty가 들어온다면 바뀌어야할 부분
    ## json 으로 item + penalty가 들어가야함. dict에 "penalty" : List[str] 추가하면 됨.
    response = requests.post(url, json=item)
    result = response.json()

    ## Json 객체로 return을 해주는데 ensure_ascii = False를 해주어야 json.dumps를 할 때 한글이 깨지지 않음
    return json.dumps(result, ensure_ascii = False)

## Todo: /summary
    ## Todo: 알맞는 timeline이 넘어옴
    ## Todo: 넘어온 timeline을 BentoML server에 전달
    ## Todo: BentoML Server Summary 태우고 다시 Backend에 전달
    ## Todo: Summary output Frontend에 전달

class SummaryInput(BaseModel):
    start: str
    due: str
    content: str
    dialogue: List[str]

@app.post('/summary')
def make_summary(item : SummaryInput):

    ## bentoml url for api summary
    url = 'http://0.0.0.0:5000/summarization'

    ## requests 보내기 위해 dict 객체로 바꿔줌
    sample = dict(item)

    ## bentoml/summarization에 summary output을 요청함
    response = requests.post(url, json=sample)
    result = response.json()

    ## 받아온 result를 Json format으로 Frontend로 보냄
    return json.dumps(result, ensure_ascii = False)

if __name__ == "__main__":
    import uvicorn

    client = MongoClient("")
    user_db = client['User'] # Cluster0라는 이름의 데이터베이스에 접속
    chat_db = client["Chat"]

    uvicorn.run(app, host="127.0.0.1", port=30001)