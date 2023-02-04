import streamlit as st
import requests
import json

st.title('Test code')

DTS_input = {
  "chat_room": "KakaoTalk_Chat_IT개발자 구직_채용 정보교류방",
  "start_date": "2023-01-11",
  "time_period": "1",
  "penalty": ["something","something2"]
}

#print(json.dumps(chat_name))
response = requests.post("http://localhost:30001/dts", json = DTS_input)

st.write("-----------------DTS Output---------------------------")
# [{"start": "2023-01-11 01:42:22", "due": "2023-01-11 08:17:14", "content": "여기", "dialogue": ""}]
st.write(response.json())

sample = json.loads(response.json())
st.write("--------------------------------------------")

st.write("-----------------Summary input---------------------------")
st.write(sample[0])

# [{"start": "2023-01-11 01:42:22", "due": "2023-01-11 08:17:14", "content": "여기", "dialogue": ""}]
response2 = requests.post("http://localhost:30001/summary", json = sample[0])

st.write("---------------------------summary output--------------------------------")
st.write(response2.json())