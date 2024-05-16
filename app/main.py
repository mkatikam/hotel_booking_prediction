# import jol
# from fastapi import FastAPI


# app = FastAPI()



# with open('app/rfc_model.pkl', 'rb') as f:
#     reloaded_model = dill.load(f)

# app = FastAPI()

# @app.get('/')
# def read_root():
#     return {"Hello": "World"}

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from joblib import load

app = FastAPI()

reloaded_model = load('app/rfc_model.pkl')


class Payload(BaseModel): 
    no_of_adults: int
    no_of_children: float
    no_of_weekend_nights: int
    no_of_week_nights: int
    required_car_parking_space: int
    lead_time: int
    arrival_year: int
    arrival_month: int
    arrival_date: int
    repeated_guest:float
    no_of_previous_cancellations:int
    no_of_previous_bookings_not_canceled:int
    avg_price_per_room:float
    no_of_special_requests:float
    type_of_meal_plan: str  # Ensure these values match the training data encoding
    room_type_reserved: str  # Ensure these values match the training data encoding
    market_segment_type: str       # Ensure these values match the training data encoding

@app.post('/')
def predict(payload: Payload):
    df = pd.DataFrame([payload.model_dump().values()], columns=payload.model_dump().keys())
    # df["booking_status"] = pd.cut(df["booking_status"],
    #                            bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
    #                            labels=[1, 2, 3, 4, 5])
    y_hat = reloaded_model.predict(df)
    return {"prediction": y_hat[0]}