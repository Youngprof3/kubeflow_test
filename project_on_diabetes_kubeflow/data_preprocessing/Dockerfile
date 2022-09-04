FROM python:latest 

WORKDIR /data_preprocessing

RUN pip install pandas sklearn numpy argparse 


COPY data_preprocessing.py ./data_preprocessing


CMD [ "python","./data_preprocessing.py"]