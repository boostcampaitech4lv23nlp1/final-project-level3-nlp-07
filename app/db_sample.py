import streamlit as st
from pymongo import MongoClient
import requests
import json

st.set_page_config(layout="wide")

def main(chat):
    st.selectbox('채팅방 선택',chat)
    
if __name__ == '__main__':
    root_password = "password"
    match = False
    placeholder = st.empty()
    with placeholder.form(key='my_form'):
        user_id = st.text_input('user_id')
        password = st.text_input('password', type="password")
        submit = st.form_submit_button(label='제출') # True or False
        if submit:
            # TODO : back으로 옮기기
            id_password = {"id":user_id,"password":password}
            response = requests.post("http://127.0.0.1:30001/login", data=json.dumps(id_password))
            if response.text.lower() == 'true':
                match = True
    if match:
        root_password = "password"
        placeholder.empty()
        response = requests.post("http://127.0.0.1:30001/chatlist", data=json.dumps(id_password))
        chat = eval(response.text)
        main(chat)