import streamlit as st
from pymongo import MongoClient
import requests

st.set_page_config(layout="wide")

def main(chat):
    st.selectbox('채팅방 선택',chat)
    
if __name__ == '__main__':
    root_password = "password"
    placeholder = st.empty()
    with placeholder.form(key='my_form'):
        user_id = st.text_input('user_id')
        password = st.text_input('password', type="password")
        submit = st.form_submit_button(label='제출') # True or False
        if submit:
            # TODO : back으로 옮기기
            response = requests.get(f"http://127.0.0.1:30001/login/{user_id}")
            root_password = eval(response.text)
    if root_password == password:
        root_password = "password"
        placeholder.empty()
        response = requests.get(f"http://127.0.0.1:30001/chatlist/{user_id}")
        chat = eval(response.text)
        main(chat)