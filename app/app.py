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
bentoml_path = '/opt/ml/bentoml/repository/SummaryService/20230130082647_CB9E5E'
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
    df = df.reset_index()
    sample = get_now(start_date,time_period, df)   # data 크기 감소
    return sample


if __name__ == '__main__':
    st.title("오픈 채팅방 요약 서비스")
    with st.form(key='my_form'):
        c1,c2 = st.columns(2)
        with c1 : start_date = st.date_input('대화 시작 시점을 설정해주세요.')
        with c2 : time_period  = st.slider('총 기간을 설정해주세요. 최대 10일입니다.', 1,10)
        uploaded_file = st.file_uploader('CSV 또는 TXT 파일을 제출해주세요.',type = ['csv', 'txt'])
        submit = st.form_submit_button(label='제출') # True or False
    # items = []
    if submit:
        # tokenizer의 경우 hash가 불가했음, unknown object type
        sample = form_return(uploaded_file, start_date, time_period)
        if len(sample) <100:
            st.warning('데이터가 충분하지 않습니다. 대화 시점과 기간을 확인해주세요.')
        with st.spinner('DTS 추론 중..'): # with 아래 까지 실행되는 동안 동그라미를 띄운다.
            # api로 날릴수 있는 부분 -> backend에 요청할 부분!
            items = bento_svc.dts(sample)
            st.success('...완료') #질문!
            # if 'items' not in st.session_state:
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
            with st.spinner('요약 생성 중..'):
                response = bento_svc.summarization(timeline)
                tab1, tab2 = st.tabs(["요약 결과", "검색 링크"])
                summary = tab1.text_area('기다려주셔서 감사합니다.',f'''
'{timeline['content']}'에 관한 {timeline['start'][:-3]}부터 {timeline['due'][:-3]}까지의 채팅 요약입니다.

{response}


YOUMbora는 당신의 채팅방이 더욱 원활하게 활용될 수 있도록 노력하고 있습니다. 
            '''
            ,height = 300)
                tab1.download_button('요약문 다운로드', summary)
                tab2.markdown(f"[키워드 검색 링크](http://google.com/search?q={timeline['content']})")
                tab2.markdown(f"[요약 검색 링크](http://google.com/search?q={response.replace(' ','')})")
                # tab2.download_button('Dows', summary)
            with cls[0]:
                # TODO : html로 구현 시 bar를 넣어서 위아래로 확인할 수 있도록
                with st.expander("원문 대화 보기"):
                    for idx, item in enumerate(timeline['dialogue'][:7]):
                        message(item, key = f"<uniquevalueofsomesort{idx}>")
# TODO : REQUEST