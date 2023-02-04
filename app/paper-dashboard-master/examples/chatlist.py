from fastapi import FastAPI
from pydantic import BaseModel, Field
from pymongo import MongoClient
from uuid import UUID, uuid4
from typing import List, Union, Optional, Dict, Any
from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi import APIRouter, FastAPI, Form, Request, requests
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.responses import JSONResponse
import uvicorn

client = MongoClient("mongodb+srv://superadmin:0214@cluster0.s2f3a.mongodb.net/test")

user_db = client['User'] # Cluster0라는 이름의 데이터베이스에 접속
chat_db = client["Chat"]

router = APIRouter(prefix="/sample")
templates = Jinja2Templates(directory='./')

class User(BaseModel):
    id : str=Form(...)
    # password : Optional[str] = None
    password: str=Form(...)


@router.get('/', response_class=HTMLResponse)
def get_login_form(request: Request):
    return templates.TemplateResponse('dashboard.html', context={'request':request})

print('333333333333333')
result = {}
chats = []
a  = user_db['user_chat_join'].find({'user_id':"jaeuk"})
for i in a:
    chats.append(i['chat_id'])
result['chat'] = chats
print(result)

@router.post('/chat', response_class=User, description='채팅방 정보를 가져옵니다.')
def get_chatlist(item : User):
    print(1)
    result = {}
    chat = []
    for d in user_db['user_chat_join'].find({'user_id':"jaeuk"}):
        chat.append(d["chat_id"])
    result['chat'] = chat
    return JSONResponse(result)
if __name__ =='__main__':
    uvicorn.run(router, host="127.0.0.1", port=30002)