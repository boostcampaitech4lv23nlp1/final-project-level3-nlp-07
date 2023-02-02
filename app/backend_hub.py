from fastapi import FastAPI
from pydantic import BaseModel, Field
from pymongo import MongoClient
from typing import List, Union, Optional, Dict, Any

app = FastAPI()

@app.get('/login')
def match_password():
    return "password"

@app.get('/login/{user_id}')
def match_password(user_id:str):
    client = MongoClient("")
    user_db = client['User'] # Cluster0라는 이름의 데이터베이스에 접속
    root_password = list(user_db['users'].find({'user_id':user_id}))[0]["password"]
    return root_password

@app.get('/chatlist/{user_id}')
def get_chatlist(user_id:str):
    chat = []
    client = MongoClient("")
    user_db = client['User'] # Cluster0라는 이름의 데이터베이스에 접속
    chat_db = client["Chat"]
    for d in user_db['user_chat_join'].find({'user_id':user_id}):
        chat.append(d["chat_id"])
    return chat
    

# if __name__ == "__main__":
#     import uvicorn

#     client = MongoClient("")
#     user_db = client['User'] # Cluster0라는 이름의 데이터베이스에 접속
#     chat_db = client["Chat"]

#     uvicorn.run("app.main:app", host="127.0.0.1", port=30001, reload=True)
