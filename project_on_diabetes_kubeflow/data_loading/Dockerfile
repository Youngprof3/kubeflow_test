FROM python:latest 

WORKDIR /data_loading

RUN pip install pandas argparse 

COPY dataset.csv ./data_loading

ADD data_loading.py ./data_loading

COPY . /data_loading/

CMD [ "python","dataset.csv","./data_loading.py" ]
