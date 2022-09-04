
FROM python:latest 

WORKDIR /model_testing

RUN pip install pandas sklearn numpy argparse 


COPY model_testing.py ./model_testing


CMD [ "python","./model_testing.py"]