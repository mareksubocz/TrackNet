FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

WORKDIR /app
#TODO odpalanie modelu z poziomu dockera, sama inferencja
# RUN /usr/local/bin/python -m pip install --upgrade pip
# RUN pip install --no-cache-dir --upgrade -r /app/TrackNet/requirements.txt

RUN /usr/local/bin/python -m pip install streamlit

EXPOSE 8501
EXPOSE 8000

COPY ./startup.sh ./

CMD ["bash", "startup.sh"]