from fastapi import FastAPI
from pymongo import MongoClient
from uuid import UUID, uuid4
import pandas as pd 
import json
from datetime import datetime, timedelta
import requests
from starlette.middleware.cors import CORSMiddleware
from prediction import total_key_word_extraction
from validation import *
app = FastAPI()
bento_API = 'http://127.0.0.1:60803'
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post('/login', description="user 정보를 가져옵니다.")
def match_password(item : User):
    root_password = user_db['users'].find_one({'user_id':item.user_id})['password']
    if root_password == item.password:
        return {"result" : 1}
    return {"result" : 0}

@app.post('/chatlist', description='채팅방 정보를 가져옵니다.')
def get_chatlist(item : User):
    chat = []
    for d in user_db['user_chat_join'].find({'user_id':item.user_id}):
        chat.append(d["chat_id"])
    print(f'유저의 가입된 이력을 바탕으로 채팅방 목록을 수집했습니다. \n {chat}')
    return {"result" : chat}

# 기간 설정이 여기 들어갈 수 있습니다.
# @app.post('/messages', description='채팅을 가져옵니다.')
def get_chattings(item):
    chat = []
    print('채팅방을 받았습니다.',item)
    print('DB에 연결되었습니다....')
    for d in chat_db[item['chat_room']].find():
        chat.append({"Date":d['Date'],"User":d['User'],"Message":d['Message']})
    print('채팅방 연결을 완료했습니다.')
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


@app.post('/keywords')
def make_keywords(item : DtsInput):
    
    chatroom = item.chat_room
    chat_dict = {"chat_room": chatroom}
    
    ## chat_dict를 활용하여 DB에 접근해 chattings를 가져옴
    chat = get_chattings(chat_dict)
    chat_df = pd.DataFrame(chat).reset_index()
    chat_message = chat_df["Message"].tolist()

    result = total_key_word_extraction(chat_message, item.penalty)

    return {'result' : result}



@app.post('/dts')
def make_dts(item : DtsInput):
    print('-'*20, "전체 대화방 분석을 시작합니다.",'-'*20)
    print(f'User Select Option : {item}')
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
    print('전처리를 완료했습니다.')

    ## JSON 형태로 넘기기 위해 datetime -> str으로 바꿔주고 / DataFrame을 dict로 변환
    sample['Date'] = sample['Date'].apply(str)
    sample_dict = sample.to_dict()
    # print(sample_dict)
    sample_dict["penalty"] = item.penalty
    ## bentoml url for api dts


    
    ## 추후 penalty가 들어온다면 바뀌어야할 부분
    ## json 으로 item + penalty가 들어가야함. dict에 "penalty" : List[str] 추가하면 됨.
    # print('sample_dict', sample_dict)
    print('전처리 정보를 토대로 Model Hub에 추론을 요청합니다.')
    response = requests.post(bento_API+'/dts', json=sample_dict)
    result = response.json() # response body가 나오는 것 -> 하지만 정확히 어떻게 변환 되는 지는 모름
    # [[dict,dict,dict,...,dict],int]
    output = {'timeline' : result[0], 'total_len' : result[1]}
    print('주제 분석을 마쳤습니다. 클라이언트로 정보를 전달하겠습니다.')
    # print('result ' , result)

    ## Json 객체로 return을 해주는데 ensure_ascii = False를 해주어야 json.dumps를 할 때 한글이 깨지지 않음
    return json.dumps(output, ensure_ascii = False)



@app.post('/summary')
def make_summary(item : SummaryInput):
    print('-'*20, "입력 받은 시점으로 요약을 시작합니다.",'-'*20)
    ## bentoml url for api summary


    ## requests 보내기 위해 dict 객체로 바꿔줌
    sample = dict(item)

    ## bentoml/summarization에 summary output을 요청함
    response = requests.post(bento_API+'/summarization', json=sample)
    result = response.json()
    print('입력받은 대화 쌍 요약을 완료했습니다.')
    ## 받아온 result를 Json format으로 Frontend로 보냄
    return json.dumps(result, ensure_ascii = False)

if __name__ == "__main__":
    import uvicorn

    client = MongoClient("mongodb+srv://superadmin:0214@cluster0.s2f3a.mongodb.net/test")
    user_db = client['User'] # Cluster0라는 이름의 데이터베이스에 접속
    chat_db = client["Chat"]

    uvicorn.run(app, host="0.0.0.0",port=30001)