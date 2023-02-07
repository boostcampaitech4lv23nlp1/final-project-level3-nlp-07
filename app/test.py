from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel, Field

class Item(BaseModel):
    items : str
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/test')
def test(item : Item):
    return {'ehkk' : item.items}

if __name__=='__main__':
    uvicorn.run(app, host="0.0.0.0",port=30002)