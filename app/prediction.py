from model import CSModel
import torch
from transformers import Automodel, BertModel
import yaml
import streamlit as st
import os
import openai

@st.cache
def load_config():
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config
@st.cache
def load_DTS(config) -> CSModel:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertModel.from_pretrained(config.DTS.BERT_PATH).to(device)
    return model
@st.cache
def load_CSmodel(config) -> CSModel:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CSModel()
    model.load_state_dict(torch.load(config.DTS.cs_PATH)).to(device)
    return model
@st.cache
def load_summary(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_prediction(model,inputs) :
    outputs = model.forward(inputs)
    return outputs

def predict_summary(config, input):
  openai.api_key = os.getenv("OPENAI_API_KEY")
  openai.api_key = config.summary.API_KEY
  response = openai.Completion.create(
    model="text-davinci-003",
    prompt='Summarize this for a second-grade student:' + input,
    temperature=0.7,
    max_tokens=64,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
  )
  return response