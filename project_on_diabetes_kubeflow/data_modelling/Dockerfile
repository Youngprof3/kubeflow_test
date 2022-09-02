FROM python:latest 

WORKDIR /data_modelling

RUN pip install pandas scikit-learn numpy argparse os


COPY data_modelling.py ./data_modelling


CMD [ "python","./data_modelling.py"]