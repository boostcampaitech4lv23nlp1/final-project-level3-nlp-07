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
from starlette.middleware.cors import CORSMiddleware
from prediction import total_key_word_extraction
app = FastAPI()
origins = [
    'http://localhost:54131',
    "http://localhost:5500",
    "http://localhost:44242",
    "http://localhost:30001",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

## Todo: Backend

## Todo: FastAPI Server 띄우기

# app = FastAPI()

## Todo: /login
    ## Todo: Frontend에서 login 정보가 넘어옴
    ## Todo: login 정보대로 DB에 접근, 카톡방 목록 가져옴
    ## Todo: 카톡방 목록을 frontend에 전달
class ForTest(BaseModel):
    data : str
@app.post('/simple')
def simple(item : ForTest):
    print('item, simple test is passed',item)
    return {'hello' :'world'}
@app.get('/test')
def teste():
    return json.dumps({'hello' :'world'},ensure_ascii = False)

if __name__ == "__main__":
    import uvicorn

    client = MongoClient("mongodb+srv://superadmin:0214@cluster0.s2f3a.mongodb.net/test")
    user_db = client['User'] # Cluster0라는 이름의 데이터베이스에 접속
    chat_db = client["Chat"]

    uvicorn.run(app, host="0.0.0.0",port=30001)