from fastapi import FastAPI
from pydantic import BaseModel, Field
from pymongo import MongoClient
from uuid import UUID, uuid4
from typing import List, Union, Optional, Dict, Any
from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory='./')

class User(BaseModel):
    id : str=Form(...)
    # password : Optional[str] = None
    password: str=Form(...)

@app.get('/login')
def get_login_form(request: Request):
    return templates.TemplateResponse('insert_username.html', 
                            context={'request':request})
# /login을 입력하면 ex.html과 연결하는 것

@app.post('/login', description="user 정보를 가져옵니다.")
def match_password(item : User):
    root_password = user_db['users'].find_one({'user_id':item.id})['password']
    if root_password == item.password:
        return True
    return False

@app.get('/chatlist')
def get_login_form(request: Request):
    return templates.TemplateResponse('dashboard.html', 
                            context={'request':request})
# /login을 입력하면 ex.html과 연결하는 것

@app.post('/chatlist', description='채팅방 정보를 가져옵니다.')
def get_chatlist(item : User):
    chat = []
    for d in user_db['user_chat_join'].find({'user_id':item.id}):
        chat.append(d["chat_id"])
    return chat
    
    
if __name__ == "__main__":
    import uvicorn

    client = MongoClient("mongodb+srv://superadmin:0214@cluster0.s2f3a.mongodb.net/test")
    user_db = client['User'] # Cluster0라는 이름의 데이터베이스에 접속
    chat_db = client["Chat"]

    uvicorn.run(app, host="127.0.0.1", port=30002)