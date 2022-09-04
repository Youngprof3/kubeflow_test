FROM python:latest 

WORKDIR /model_evaluation

RUN pip install pandas sklearn numpy argparse 


COPY model_evaluation.py ./model_evaluation


CMD [ "python","./model_evaluation.py"]