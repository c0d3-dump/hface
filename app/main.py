from .model import translate
from typing import Union
from pydantic import BaseModel

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


class Item(BaseModel):
    input: str
    lang_from: str
    lang_to: str


@app.post("/translate")
def tramslatee(item: Item):
    output = translate(item.input, src=item.lang_from, tgt=item.lang_to)
    return {"output": output}
