from pydantic import BaseModel, Field
from typing import List, Union, Optional, Dict, Any
class User(BaseModel):
    user_id : Optional[str] = None
    password : Optional[str] = None

class Chatlist(BaseModel):
    chat_room : str

class Chat(BaseModel):
    Date: str
    User: str
    Message: str

class DtsInput(BaseModel):
    chat_room : str
    start_date : str ## or datetime, 아마 string일 듯
    time_period : str
    penalty : List[str]

class SummaryInput(BaseModel):
    dialogue: List[str]