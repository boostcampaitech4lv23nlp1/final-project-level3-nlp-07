import streamlit as st
from streamlit_timeline import st_timeline
from streamlit_chat import message
import pandas as pd
import requests
import re
import chardet
from datetime import timedelta
from prediction import *
import bentoml


st.set_page_config(layout="wide")

root = 'http://0.0.0.0:8001/'
bentoml_path = '/opt/ml/bentoml/repository/SummaryService/20230129175416_E78DE5'
bento_svc = bentoml.load(bentoml_path)

def get_now(start_date, time_period, df):    
    df['Date'] = pd.to_datetime(df['Date'],infer_datetime_format=True)
    sample = df[df['Date'].isin(pd.date_range(str(start_date), str(start_date + timedelta(days=time_period)),freq = 's'))]
    return sample

def txt_to_csv(uploaded_file, encoding):
    ymd_format = '\d{4}년 \d{1,2}월 \d{1,2}일'
    raw_data = []
    for r in uploaded_file.getvalue().decode(encoding).splitlines():
        raw_date = re.findall(ymd_format, r)
        if len(raw_date)>0:
            idx_date='-'.join([d if len(d)>1 else '0'+d for d in re.findall('\d+',raw_date[0])])
        else:
            raw_sentence = r.replace('\n','').replace('[','').split(']')
            if len(raw_sentence)>1:
                try:
                    pmam, hm = raw_sentence[1].lstrip().split()
                    if pmam == '오전':
                        pmam = 'AM'
                    else:
                        pmam = 'PM'
                    hm = hm +':00'
                    if len(hm)<8:
                        hm = '0'+hm
                    fin_date=' '.join([idx_date,hm,pmam])
                    raw_data.append([fin_date,raw_sentence[0].strip(),raw_sentence[2].lstrip()])
                except:
                    pass
    fin_pd = pd.DataFrame(raw_data,columns=['Date','User','Message'])
    return fin_pd

def form_return(uploaded_file, start_date, time_period):
    # chardet 라이브러리로 인코딩 확인 후 맞춰서 encoding, csv와 txt의 getvalue() 형식이 다르다.
    if uploaded_file.name.split('.')[-1]=='csv':
        encoder_type = chardet.detect(uploaded_file.getvalue())['encoding']
        df = pd.read_csv(uploaded_file, encoding=encoder_type) # 2가지 케이스만 한거라 에러를 만들어 보는걸 추천
    else:
        encoder_type = chardet.detect(uploaded_file.getvalue().splitlines()[0])['encoding']
        df = txt_to_csv(uploaded_file, encoding=encoder_type)
    sample = get_now(start_date,time_period, df)   # data 크기 감소
    return sample


if __name__ == '__main__':
    st.title("Open Talk Demo")
    with st.form(key='my_form'):
        c1,c2 = st.columns(2)
        with c1 : start_date = st.date_input('this start from...')
        with c2 : time_period  = st.slider('Max day you can check is 10 days', 1,10)
        uploaded_file = st.file_uploader('upload your OpenTalk csv or txt',type = ['csv', 'txt'])
        submit = st.form_submit_button(label='Submit') # True or False
    # items = []
    if submit:
        # tokenizer의 경우 hash가 불가했음, unknown object type
        sample = form_return(uploaded_file, start_date, time_period)
        if len(sample) <100:
            st.warning('Too Small data to summary... please update time_period and start_date')
        with st.spinner('get_DTS..'): # with 아래 까지 실행되는 동안 동그라미를 띄운다.
            items = bento_svc.dts(sample)
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
        # else:
        #     st.warning("items not available")
        if timeline is not None:
            with st.spinner('get_summary..'):
                response = bento_svc.summarization(timeline)
                tab1, tab2 = st.tabs(["Summary_output", "Viz"])
                summary = tab1.text_area('summary Output',f'''
{response}

YEOMbora에서는 당신의 채팅방을 더욱 원활하게 활용될 수 있도록 노력하고 있습니다. 
            '''
            ,height = 300)
                tab1.download_button('Download summary text', summary)
                tab2.title('시각화 대상') # 시각화 추가 
                tab2.download_button('Dows', summary)
            with cls[0]:
                # TODO : html로 구현 시 bar를 넣어서 위아래로 확인할 수 있도록
                with st.expander("대화보기 -> 궁금하냐? ㅋ"):
                    for idx, item in enumerate(timeline['dialogue'][:7]):
                        message(item, key = f"<uniquevalueofsomesort{idx}>")
# TODO : REQUEST