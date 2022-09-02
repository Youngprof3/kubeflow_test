FROM python:latest 

WORKDIR /data_loading

RUN pip install pandas argparse os

COPY data_loading.py ./data_loading

ADD dataset_of_diabetes.csv ./data_loading

COPY . /data_loading/

CMD [ "python","./data_loading.py","dataset_of_diabetes.csv" ]