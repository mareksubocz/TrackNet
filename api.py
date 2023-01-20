from fastapi import FastAPI
from pydantic import BaseModel

# TODO możliwe ze ścieżki
# from TrackNet import TrackNet

app = FastAPI()

tracknetPath = ""  # TODO zapisanie jakoś modelu


# trackNet = TrackNet()
# trackNet.load(path=tracknetPath, device="CUDA:0")


# TODO jak odpytywać zapisany model??
# w streamlicie odczytam bajty,
# wiec pewnie trzeba to jakos spowrotem na tablice odpowiednią sparsować
class Input(BaseModel):
    video: bytes
    text: str


class Prediction(BaseModel):
    result: bytes


# @app.post("/predict", response_model=Prediction)
# def predict(payload: Input):
#     return trackNet.forward(payload.video)


@app.get("/")
def home():
    return {
        "refresh": "OK",
        "health check": "OK",
        "model_version": 0
    }  # TODO wersjonowanie modelu
