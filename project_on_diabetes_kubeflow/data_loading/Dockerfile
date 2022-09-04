FROM python:latest 

WORKDIR /data_loading

RUN pip install pandas argparse requests

COPY data_loading.py ./data_loading

COPY . /data_loading/

CMD [ "python","./data_loading.py" ]
