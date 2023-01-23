import streamlit as st
from streamlit_timeline import st_timeline
from streamlit_chat import message
import pandas as pd
from omegaconf import OmegaConf
import requests
import time
import argparse
from transformers import AutoTokenizer
from datetime import datetime, timedelta
from prediction import *

st.set_page_config(layout="wide")

root = 'http://0.0.0.0:8001/'

def get_now(start_date, time_period, df):    
    df['Date'] = pd.to_datetime(df['Date'],infer_datetime_format=True)
    sample = df[df['Date'].isin(pd.date_range(str(start_date), str(start_date + timedelta(days=time_period)),freq = 's'))]
    return sample
def main():
    st.title("Open Talk Demo")
    # TODO : slider
    c1,c2 = st.columns(2)
    with c2 :time_period  = st.slider('Max day you can check is 10 days', 1,10)
    with c1 : start_date = st.date_input('this start from...')
    uploaded_file = st.file_uploader('upload your OpenTalk csv',type = {'csv'})
    if uploaded_file:
        df = pd.read_csv(uploaded_file,encoding='cp949')
        sample = get_now(start_date,time_period, df)

        # data = json.dumps(data)
        # response = requests.post(root + 'upload', json=data)
        # response = requests.post(root + 'text', files = files)
        st.dataframe(sample)
        return uploaded_file
    # TODO : file upload
    # TODO : range check 원하는 일자 -> 
    # TODO : 확인버튼
def show(upload_files,cs_model, bert_model, tokenizer,cfg):
    col1, _,col2 = st.columns([0.27,0.03,0.7],gap ='small')
    with col2:
        if 'DTS' not in st.session_state:
            with st.spinner(text='In progress'):
                # must be fastapi
                label = get_DTS(bert_model=bert_model, cs_model=cs_model, tokenizer=tokenizer, inputs = upload_files) 
                items = get_timeline(upload_files, label)
                # implement by fastapi after
                st.session_state.DTS = True
                st.success('Done')
                timeline = st_timeline(items, groups=[], options={}, height="300px")             # DTS 시각화
                st.subheader("Selected item")
        if timeline:
            with st.spinner('please wait.. now summarization'):
                # must be fastapi
                does = predict_summary(cfg, inputs = items)
                if does : st.success('summary ... done')
                # implement by fastapi after
                # TODO : inference time check 필요
            tab1, tab2 = st.tabs(["Summary_output", "Viz"])
            summary = tab1.text_area('summary Output',f'''

{timeline['summary']}

YEOMbora에서는 당신의 채팅방을 더욱 원활하게 활용될 수 있도록 노력하고 있습니다. 
            '''
            ,height = 300)
            tab1.download_button('Download summary text', summary)
            tab2.title('시각화 대상')
            tab2.download_button('Dows', summary)
    with col1: 
        with st.expander("대화보기 -> 궁금하냐? ㅋ"):
            if timeline:
                for idx, item in enumerate(timeline['dialouge']):
                    message(item,key = f"<uniquevalueofsomesort{idx}>")

                message("Hello bot!", is_user=True)  # align's the message to the right  

if __name__ == '__main__':
    parsers = argparse.ArgumentParser()
    parsers.add_argument('--config', type=str, default='config')
    args, _ = parsers.parse_known_args()
    cfg = OmegaConf.load(f'./config/{args.config}.yaml')
    cs_model = load_CSmodel(cfg)
    bert_model = load_DTS(cfg)
    tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
    upload_files = main()
    if upload_files:
        show(upload_files)