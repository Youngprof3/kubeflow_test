FROM python:latest 

WORKDIR /data_preprocessing

RUN pip install pandas scikit-learn numpy argparse os


COPY data_preprocessing.py ./data_preprocessing


CMD [ "python","./data_preprocessing.py"]