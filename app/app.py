import streamlit as st
from streamlit_timeline import st_timeline
from streamlit_chat import message
import pandas as pd
from omegaconf import OmegaConf
import requests
import time
import argparse
from datetime import datetime, timedelta

st.set_page_config(layout="wide")

root = 'http://0.0.0.0:8001/'
a = ['''채용공고 게시 규칙을 모두 만족합니다 

스마일게이트 스토브 신규서비스 백엔드, 프론트엔드 모집중입니다

백엔드 공고
https://careers.smilegate.com/apply/announce/view?seq=4330

프론트엔드 공고
https://careers.smilegate.com/apply/announce/view?seq=4283''',
'오 스마게',
'이모티콘',
'많관부~',
'지금 지원하시면 내년 상반기에 무료로 여행을 보내줍니다! 쿄쿄쿄',
'오옹?',
'지원해서 합격하면 무료여행이라니 지역 상관없느긴가용',
'20주년이라고 각 팀마다 여행을 보내주는게 있어요',
'아직 여행 안간 팀이어서 내년 6월 이전에 한번 보내준답니다!',
'예산 빵빵하게!',
'샵검색 : #스마일게이트 플레이숍',
'와 스마게',
'갓스마게 ㄷㄷ',
'스마일게이트..?',
'이모티콘',
'경력이라',
'이모티콘',
'0.5년차인데',
'6배 쳐줍니까',
'연봉/6 해드립니다.']
# TODO : DTS 아웃풋(binary_label)을 활용해 Dict파일을 만든다 ->자동화 필요 
items = [
    {"id": 1, "content": "정치", "start": "2022-10-20", 'summary' : 'this is summary', 'dialouge' : a}, # content : top keyword
    {"id": 2, "content": "IT 취업", "start": "2022-10-09", 'summary' : 'this is summary', 'dialouge' : a},
    {"id": 3, "content": "코딩 테스트", "start": "2022-10-18", 'summary' : 'this is summary', 'dialouge' : a},
    {"id": 4, "content": "대학교 생활", "start": "2022-10-16",'summary' : 'this is summary', 'dialouge' : a},
    {"id": 5, "content": "인생", "start": "2022-10-25", 'summary' : 'this is summary', 'dialouge' : a},
    {"id": 6, "content": "현실", "start": "2022-10-27", 'summary' : 'this is summary', 'dialouge' : a},
]
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
        data = sample.to_json()
        # data = json.dumps(data)
        # response = requests.post(root + 'upload', json=data)
        # response = requests.post(root + 'text', files = files)
        st.dataframe(sample)
        return uploaded_file
    # TODO : file upload
    # TODO : range check 원하는 일자 -> 
    # TODO : 확인버튼
def show(upload_files):
    col1, _,col2 = st.columns([0.27,0.03,0.7],gap ='small')
    with col2:
        if 'DTS' not in st.session_state:
            with st.spinner(text='In progress'):
                time.sleep(5)
                st.session_state.DTS = True
                st.success('Done')
                timeline = st_timeline(items, groups=[], options={}, height="300px")             # DTS 시각화
                st.subheader("Selected item")
        if timeline:
            with st.spinner('please wait.. now summarization'):
                time.sleep(5)
                tab1, tab2 = st.tabs(["Summary_output", "Viz"])
                    # TODO : inference Summary model
                # TODO : sum = summary model(a)
                # TODO : inference time check 필요
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
    upload_files = main()
    if upload_files:
        show(upload_files)