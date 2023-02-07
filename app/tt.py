from fastapi import FastAPI, Request, Response
from pydantic import BaseModel, Field
from pymongo import MongoClient
from uuid import UUID, uuid4
from typing import List, Union, Optional, Dict, Any
import pandas as pd 
import json
from datetime import datetime, timedelta
import time
import requests
from starlette.middleware.cors import CORSMiddleware
from prediction import total_key_word_extraction
from fastapi.encoders import jsonable_encoder
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DtsInput(BaseModel):
    chat_room : str
    start_date : str ## or datetime, 아마 string일 듯
    time_period : str
    penalty : List[str]

def get_chattings(item):
    chat = []
    print('in_chat item',item)
    print('db Key check',chat_db[item['chat_room']])
    for d in chat_db[item['chat_room']].find():
        chat.append({"Date":d['Date'],"User":d['User'],"Message":d['Message']})
    print('끝난 chat', chat)
    return chat

def get_now(start_date, time_period, df):
    print(df) 
    df['Date'] = pd.to_datetime(df['Date'],infer_datetime_format=True)
    sample = df[df['Date'].isin(pd.date_range(str(start_date), str(start_date + timedelta(days=time_period)),freq = 's'))]
    return sample
    
from typing import Dict
@app.post('/dts')
def make_dts(request):
    print(request)
    # item = jsonable_encoder(item)
    return request
    # item = json.loads(item, encoding="utf-8")

    # chatroom = item['chat_room']
    # chat_dict = {"chat_room": chatroom}
    
    # ## chat_dict를 활용하여 DB에 접근해 chattings를 가져옴
    # chat = get_chattings(chat_dict)

    # ## 가져온 chattings를 DataFrame으로 변환하고 index 만들어줌 (그냥은 index가 없는데 추후 get_DTS에 필요함)
    # chat_df = pd.DataFrame(chat).reset_index()
    
    # ## str으로 넘어온 start_date를 datetime class로 바꿔줌
    # temp = item['start_date']
    # timestamp = datetime.strptime(temp, '%Y-%m-%d') ## 임시 : str to datetime
    # # '{"chat_room":"IT 개발자 구직 채용 정보교류방","start_date":"2022-12-16","time_period":"10","penalty":["AI","ML"]}'
    # ## get_now를 통과하여 start_date ~ start_date + time_period 까지 sampling 진행
    # sample = get_now(timestamp, int(item['time_period']), chat_df)
    # print('sample', sample)

    # ## JSON 형태로 넘기기 위해 datetime -> str으로 바꿔주고 / DataFrame을 dict로 변환
    # sample['Date'] = sample['Date'].apply(str)
    # sample_dict = sample.to_dict()
    # # print(sample_dict)
    # sample_dict["penalty"] = item['penalty']
    # ## bentoml url for api dts
    # url = 'http://127.0.0.1:45374/dts'

    
    # ## 추후 penalty가 들어온다면 바뀌어야할 부분
    # ## json 으로 item + penalty가 들어가야함. dict에 "penalty" : List[str] 추가하면 됨.
    # print('sample_dict', sample_dict)
    # response = requests.post(url, json=sample_dict)
    # result = response.json()
    # print('result ' , result)

    # ## Json 객체로 return을 해주는데 ensure_ascii = False를 해주어야 json.dumps를 할 때 한글이 깨지지 않음
    # return json.dumps(result, ensure_ascii = False)

from fastapi import FastAPI, Header, Depends, HTTPException
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    description: str
    price: float

@app.post("/items/")
async def create_item(item: Item):
    return {"name": item.name, "description": item.description}
if __name__ == "__main__":
    import uvicorn

    client = MongoClient("mongodb+srv://superadmin:0214@cluster0.s2f3a.mongodb.net/test")
    user_db = client['User'] # Cluster0라는 이름의 데이터베이스에 접속
    chat_db = client["Chat"]

    uvicorn.run(app, host="0.0.0.0",port=30002)