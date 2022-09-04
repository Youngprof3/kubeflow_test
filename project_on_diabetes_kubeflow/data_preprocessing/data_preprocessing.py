
import pandas as pd
import argparse
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def data_preprocessing(data):

    
    le = LabelEncoder()
    
    data['Gender']= data.loc[:,['Gender']].apply(le.fit_transform)
    
    data['CLASS']= data.loc[:,['CLASS']].apply(le.fit_transform)
    
    predictors_df = data.iloc[:,:-1]
    
    target_df = data.iloc[:,-1:]
    
    scaler = MinMaxScaler(feature_range=(0,1))
    
    rescaled_df = pd.DataFrame(scaler.fit_transform(predictors_df), columns= predictors_df.columns)
    
    scaler = Normalizer()
    
    normalized_df = pd.DataFrame(scaler.fit_transform(rescaled_df), columns= rescaled_df.columns)
    
    x_train, x_test, y_train, y_test = train_test_split(normalized_df,target_df, test_size=0.3, random_state=42)
    

    data_path = './data_preprocessing/preprocessed_data'
    
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    np.save(f'{data_path}/x_train.npy', x_train)  
    np.save(f'{data_path}/x_test.npy', x_test)
    np.save(f'{data_path}/y_train.npy', y_train)
    np.save(f'{data_path}/y_test.npy', y_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--x_train')
    parser.add_argument('--x_test')
    parser.add_argument('--y_train')
    parser.add_argument('--y_test')
    args = parser.parse_args()
    data_preprocessing(args.x_train, args.x_test, args.y_train, args.y_test)