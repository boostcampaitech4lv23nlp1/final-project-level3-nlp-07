import streamlit as st
from pymongo import MongoClient
import requests
import json

st.set_page_config(layout="wide")
    
if __name__ == '__main__':
    root_password = "password"
    match = False
    with st.form(key='my_form'):
        user_id = st.text_input('user_id')
        password = st.text_input('password', type="password")
        submit = st.form_submit_button(label='제출') # True or False
        if submit:
            id_password = {"id":user_id,"password":password}
            response = requests.post("http://127.0.0.1:30001/login", json=id_password)
            if response.text.lower() == 'true':
                match = True
    if match:
        root_password = "password"
        password = "block"
        response = requests.post("http://127.0.0.1:30001/chatlist", json=id_password)
        chat = eval(response.text)
        response = requests.post("http://127.0.0.1:30001/messages", json={"chat_room":chat[0]})
        st.write(response.text)