import streamlit as st
from streamlit_timeline import st_timeline
from streamlit_chat import message
import pandas as pd
from omegaconf import OmegaConf
import requests
import time
import argparse
from transformers import AutoTokenizer
from functools import partial
from datetime import datetime, timedelta
from prediction import *


st.set_page_config(layout="wide")

root = 'http://0.0.0.0:8001/'

def get_now(start_date, time_period, df):    
    df['Date'] = pd.to_datetime(df['Date'],infer_datetime_format=True)
    sample = df[df['Date'].isin(pd.date_range(str(start_date), str(start_date + timedelta(days=time_period)),freq = 's'))]
    return sample

def form_return(uploaded_file, start_date, time_period):
    # uploaded_file, start_date, time_period = args
    try:
        df = pd.read_csv(uploaded_file,encoding='cp949')
    except:
        df = pd.read_csv(uploaded_file,encoding='utf-8') # 2가지 케이스만 한거라 에러를 만들어 보는걸 추천
    sample = get_now(start_date,time_period, df)   # data 크기 감소
    parsers = argparse.ArgumentParser()
    parsers.add_argument('--config', type=str, default='config')
    args, _ = parsers.parse_known_args()
    cfg = OmegaConf.load(f'./{args.config}.yaml')
    cs_model = load_CSmodel(cfg)
    bert_model = load_DTS(cfg)
    tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
    return sample,cs_model, bert_model, tokenizer,cfg

if __name__ == '__main__':
    st.title("Open Talk Demo")
    with st.form(key='my_form'):
        c1,c2 = st.columns(2)
        with c2 : time_period  = st.slider('Max day you can check is 10 days', 1,10)
        with c1 : start_date = st.date_input('this start from...')
        uploaded_file = st.file_uploader('upload your OpenTalk csv',type = {'csv'})
        submit = st.form_submit_button(label='Submit') # True or False
    items = []
    if submit:
        sample,cs_model, bert_model, tokenizer,cfg = form_return(uploaded_file, start_date, time_period)
        if len(sample) <100:
            st.warning('Too Small data to summary... please update time_period and start_date')
        with st.spinner('get_DTS..'): # with 아래 까지 실행되는 동안 동그라미를 띄운다.
            # api로 날릴수 있는 부분 -> backend에 요청할 부분!
            items = get_DTS(bert_model=bert_model,cs_model=cs_model,tokenizer=tokenizer,inputs = sample) #pandas
            st.success('Done...') #질문!
            if 'items' not in st.session_state: 
                # dict key value -> 선언을 하게 되면 로컬변수가 아니라 캐시로 저장을 하게 됩니다.
                # with문 안에서 선언되는 것들 또는 함수 안에서 진행되는 변수들을 전역 변수로 바꿔 준다고 이해
                st.session_state['items'] = items
    cls = st.columns([0.27,0.03,0.7],gap ='small') # 화면 분할 레이어 3개로 
    with cls[2]:
        timeline = None # 지역변수니까!
        if 'items' in st.session_state: # 키를 나열해요 st.session_State = [key1, key2]
            timeline = st_timeline(st.session_state['items'], groups=[], options={}, height="300px")             # DTS 시각화
            # https://github.com/giswqs/streamlit-timeline/blob/master/streamlit_timeline/__init__.py 
        else:
            st.warning("items not available")
        if timeline is not None:
            with st.spinner('get_summary..'):
                sums = predict_summary(inputs = timeline) #전체를 하는게 아니라 하나만!
                tab1, tab2 = st.tabs(["Summary_output", "Viz"])
                summary = tab1.text_area('summary Output',f'''
{sums}

YEOMbora에서는 당신의 채팅방을 더욱 원활하게 활용될 수 있도록 노력하고 있습니다. 
            '''
            ,height = 300)
                tab1.download_button('Download summary text', summary)
                tab2.title('시각화 대상') # 시각화 추가 
                tab2.download_button('Dows', summary)
            with cls[0]:
                with st.expander("대화보기 -> 궁금하냐? ㅋ"):
                    for idx, item in enumerate(timeline['dialouge']):
                        message(item,key = f"<uniquevalueofsomesort{idx}>")
# TODO : 에러처리 특히 PANDAS
# TODO : REQUEST
# TODO : 