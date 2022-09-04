import os
import argparse
import pandas as pd

def data_loading(args):
    '''
    Function for loading diabetes dataset
    '''
    import urllib.request
    url = 'https://raw.githubusercontent.com/Youngprof3/kubeflow_test/main/project_on_diabetes_kubeflow/data_loading/dataset.csv'
    resp = urllib.request.urlopen(url)
    data = pd.read_csv(resp,na_values=['?'])
    data_path = './data_loading'

    if not os.path.exists(data_path):
        os.makedirs(data_path)
        
    with open(f'{data_path}/data_loading', 'wb') as file:
        file.write(data.content)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url')
    args = parser.parse_args()
    data_loading(args) 
    