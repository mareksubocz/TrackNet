from fastapi import FastAPI
from pydantic import BaseModel

# TODO możliwe ze ścieżki
from TrackNet import TrackNet

app = FastAPI()

tracknetPath = ""  # TODO zapisanie jakoś modelu


trackNet = TrackNet()
trackNet.load(path=tracknetPath, device="CUDA:0")


# TODO jak odpytywać zapisany model??
# w streamlicie odczytam bajty,
# wiec pewnie trzeba to jakos spowrotem na tablice odpowiednią sparsować
#filmy trafiają w formacie mp4

"""
   uzytkownik wrzuca filmik np 30min
   
   póki co: folder /tmp i elo

    1 mecz do 2BB
    1s +/- 1MB
   
   docelowo ->

    Błażej
    rozkminić inferencje batchową

    START AKCJI
    ---analiza
    KONIEC AKCJI
    
    
    1. osobne AI do wykrywania zmiany stron
     i to działa jako triger -> ZMIANA STRONY!!!
    
    
    działanie na turnieju:
    1. Kamera z raspbery podpiętym wysyłania nagranej akcji na komputer
    2. raspberry połączone po kablu do lapka RASPBERRY ogarnia temat dzielenia na akcje
    3. podzielony filmik trafia do jakiejś kolejki - moze rabbitmq
    4. co akcje z głównego modelu od statystyk
    5. stworzyć zwykły serwis do agregacji statystyk 
    
    kroki aplikacji:
    1. STREMLIT APP - użytkownik wrzuca filmik np 8min (czyli gra po jednej stronie siatki)
    2. FAST API -> w jakiś sposób tworze kolejke klatek, które przesyłam do modelu TODO to jest ważne do ogarniecia
    3. Model -> odpytanie sieci
        a) pierwsza siec -> dzieli na akcje
        b) druga sieć -> rozpoznaje zmiane stron
        c) 
    4. wynik modelu na bieżąco akumulowany 
    5. 
    
    
    
    Marek:
        - inferencja pojedyncza klatka
    co zwraca model:
        -  
     
    Błażej:
        - odczytywać filmy w formacie [mp4]
        - wprowadzić system kolejkowania inputu do Sieci
        - agregacja wyników z sieci od zmiany stron do zmiany strong
    
"""
class Input(BaseModel):
    video: bytes
    text: str


class Prediction(BaseModel):
    result: bytes


@app.post("/predict", response_model=Prediction)
def predict(payload: Input):
    return trackNet.forward(payload.video)


@app.get("/")
def home():
    return {
        "refresh": "OK",
        "health check": "OK",
        "model_version": 0
    }  # TODO wersjonowanie modelu
