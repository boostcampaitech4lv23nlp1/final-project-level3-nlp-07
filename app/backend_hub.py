from fastapi import FastAPI
from pydantic import BaseModel, Field
from pymongo import MongoClient
from uuid import UUID, uuid4
from typing import List, Union, Optional, Dict, Any
import pandas as pd 

app = FastAPI()

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
@app.post('/messages', description='채팅을 가져옵니다.')
def get_chattings(item : Chatlist):
    chat = []
    for d in chat_db[item.chat_room].find():
        chat.append({"Date":d['Date'],"User":d['User'],"Message":d['Message']})
    return chat


if __name__ == "__main__":
    import uvicorn

    client = MongoClient("")
    user_db = client['User'] # Cluster0라는 이름의 데이터베이스에 접속
    chat_db = client["Chat"]

    uvicorn.run(app, host="127.0.0.1", port=30001)
