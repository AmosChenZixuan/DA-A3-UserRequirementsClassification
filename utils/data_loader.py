import os 
import re
import json
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .text_processor import TextProcessor as tp
from .vectorize import Vectorizer as vec
from config import SEED, SPLIT


class DataLoader:
    DIR_NAME = 'json/'
    FIELDS = ['comment', \
        'rating', 'past', 'future', 'length_words', \
        'sentiScore', \
        #'sentiScore_pos', 'sentiScore_neg', \
        'label']

    def __init__(self):
        self.raw = self._load_data()
        # preprocess
        self.df = None
        self.labels = None 
        self.preprocess(self.raw)

        # split
        self.datasets = None
        self.split_dataset(self.df)

    def get_train(self):
        return self.datasets[0], self.datasets[2]

    def get_test(self):
        return self.datasets[1], self.datasets[3]

    def split_dataset(self, df):
        X_train, X_test, y_train, y_test = \
            train_test_split(df[self.feature_names], df[self.label_name], \
                            test_size=SPLIT, random_state=SEED)

        # rescale 'length_words'
        scalar = StandardScaler()
        scalar.fit(X_train['length_words'].to_numpy().reshape(-1,1))
        X_train['length_words'] = scalar.transform(X_train['length_words'].to_numpy().reshape(-1,1))
        X_test['length_words'] = scalar.transform(X_test['length_words'].to_numpy().reshape(-1,1))

        # text feature vectorization
        train_vectorized_text, test_vectorized_text = vec.tfidf(X_train[self.text_col], X_test[self.text_col],max_df=1.)

        X_train =  np.hstack([train_vectorized_text, X_train[self.other_col].to_numpy()])
        X_test =  np.hstack([test_vectorized_text, X_test[self.other_col].to_numpy()])

        self.datasets = X_train, X_test, y_train, y_test

    def preprocess(self, df):
        df = self.feature_preprocess(df)
        df = self.label_preprocess(df)
        self.df = df 


    def feature_preprocess(self, df):
        df = self.drop_duplicates(df)
        df = self.drop_negatives(df)
        df = tp.preprocess_pipeline(df, self.text_col)
        # TODO: other features
        return df

    def label_preprocess(self, df):
        label = DataLoader.FIELDS[-1]
        le = preprocessing.LabelEncoder()
        le.fit(df[label])
        df[label] = le.transform(df[label])

        self.labels = le.classes_
        return df



    def _load_data(self):
        df = pd.DataFrame([], columns = DataLoader.FIELDS)

        dir_name = DataLoader.DIR_NAME
        for file in os.listdir(dir_name):
            data = self._load_file(os.path.join(dir_name, file))
            df = pd.concat([df, pd.DataFrame(data, columns = DataLoader.FIELDS)])
        df = df.reset_index().drop(columns=['index'])
        return df

    def _load_file(self, filename):
        with open(filename) as json_file:
            raw_data = json.load(json_file)
            return self._convert_data(raw_data)

    @staticmethod
    def _convert_data(raw_data):
        data = []
        for elem in raw_data:
            data.append([elem[field_name] for field_name in DataLoader.FIELDS])
        return data

    @staticmethod
    def drop_duplicates(df):
        features = list(df.columns[:-1])
        df['freq'] = df.groupby(features).transform('count')
        mask = [] 

        for i, row in df.iterrows():
            discard = False
            if row['freq'] > 1:
                # find duplicate rows which have all features matched
                matched = 1
                for feature in features:
                    matched &= df[feature] == row[feature] 
                duplicates = df[matched]
                # count unique labels
                # if there are more than one positive label, then it is a multi-label data point
                all_labels = duplicates['label'].unique()
                positive_count = sum([not x.startswith('Not') for x in all_labels])
                discard = positive_count > 1

            mask.append(discard)

        df = df[~np.array(mask)] 
        df = df.drop(columns=['freq'])
        return df

    @staticmethod
    def drop_negatives(df):
        mask = df['label'].str.startswith('Not')
        df = df[~mask]
        return df

    @property
    def feature_names(self):
        return self.FIELDS[:-1]

    @property
    def text_col(self):
        return self.feature_names[0]
    
    @property
    def other_col(self):
        return self.feature_names[1:]

    @property
    def label_name(self):
        return self.FIELDS[-1]