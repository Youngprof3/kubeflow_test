FROM python:latest 

WORKDIR /data_modelling

RUN pip install pandas sklearn numpy argparse 


COPY data_modelling.py ./data_modelling


CMD [ "python","./data_modelling.py"]