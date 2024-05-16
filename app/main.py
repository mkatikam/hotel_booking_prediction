from dill import load
from fastapi import FastAPI

with open('app/rfc_model.pkl','rb') as f:
    reloaded_model = load(f)

app = FastAPI()

@app.get('/')

def read_root():
    return {"Hello":"World"}