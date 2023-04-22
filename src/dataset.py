import os
from pathlib import Path
import pandas as pd
import networkx as nx
import numpy as np
from utils import Utilities as utils
from sklearn import preprocessing
from torch.utils.data import Dataset
import torch

class ToTensor(object):
    """Convert sample to Tensors."""

    def __call__(self, sample):
        return torch.Tensor(sample).float() 
    
class DataFrameEntry():
    def __init__(self, columns : list, values : list, name = '') -> None:
        self.columns = columns
        self.values = values
        self.name = name

class DataFrameLabel():
    def __init__(self, columns : list, values : list, name = '') -> None:
        self.columns = columns
        self.values = values
        self.name = name
        
    
class Dataset(Dataset):
    def __init__(self, filePath : str, label_column : list, separator = ';', name=''):
        self.dataframe = utils.createDataframe(filepath=filePath, 
                                               separator=separator)
        self.label_columns = label_column
        self.encoders = {}

    def init_label_dictionary(self, label_column, label_values):
        for column in self.label_columns:
            labels = self.dataframe[column].unique()
            self.label_dictionary[column] = {}


    def __len__(self):
        return len(self.dataframe)
    
    def get_labels(self):
        return self.df[self.label_column]

    def encode_column(self, column : str | int) -> None:
        if self.encoders.get(column) is None:
            self.encoders[column] = preprocessing.LabelEncoder()
            self.encoders[column].fit(self.dataframe[column].values)
        self.dataframe[column] = self.encoders[column].transform(self.dataframe[column].values)

    def decode_column(self, column : str | int) -> None:
        if self.encoders.get(column) is not None:
            self.dataframe[column] = self.encoders[column].inverse_transform(self.dataframe[column].values)
        else:
            print('Warning: Column not encoded')

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.dataframe.iloc[idx]
    
    def get_column_types(self, column : str | int) -> str:
        if self.__len__() > 0:
            return type(self.dataframe[column][0])
        else:
            raise Exception('Dataset is empty')
    

            
    def addDataset(self, filePath : str, separator = ';', name='') -> None:
        self.df = utils.createDataframe(self.base_path + filePath, separator=separator)
        
        if self.df is None:
            print('Error: File not found or not valid')
        else:
            if self.dataframes.get(name) is None:
                self.dataframes[name] = self.df
                self.encode_df = self.df.copy()
                print('Added ' + filePath + ' to dataset')
            else: 
                print('Warning: Dataset name already exists')
             
    
    def createDataset(self, files : list) -> None:
        frames = []
        for file in files:
            csv = self.base_path + file
            df = pd.read_csv(csv, sep=";")
            frames.append(df)
        self.df = pd.concat(frames) 
        self.df.drop_duplicates(inplace=True)

    def get_feature_count(self):
        return len(self.dataframe.columns) - 1
    
    def get_label_count(self):
        return len(self.dataframe[self.label_column].unique())
        
    def cleanDataframe(self):
        # Check for columns with all different values
        size = self.dataframe.shape
        self.dataframe = self.dataframe.loc[:, self.dataframe.apply(pd.Series.nunique) != self.dataframe.shape[0]]
        
        # Exclude some entries as to make it even
        self.dataframe = self.dataframe[:self.dataframe.shape[0] - (self.dataframe.shape[0] % 10)]
        print("Removed: " + str(size[0] - self.dataframe.shape[0]) + " rows | " + 
              str(size[1] - self.dataframe.shape[1]) + " columns")
         
    def applyPreprocessing(self, columns:list):
        size = self.df.shape[1]
        self.select(columns)
        print("Removed " + str(size - self.df.shape[1]) + " columns")
        
    def select(self, columns:list):
        if self.deleted is None:
            self.deleted = pd.DataFrame()
        
        # Restore the deleted columns
        # self.restore(columns)
                
        # Keep track of the deleted columns
        _deletedColumns = self.df.columns.difference(columns)
        
        if self.deleted.empty:
            self.deleted = self.df[_deletedColumns]
        else:
            self.deleted = pd.concat([self.deleted, self.df[_deletedColumns]], axis=0)

            
        self.df.drop(_deletedColumns, axis=1, inplace=True)

    ## NEEDS TO BE FIXED ##
    def restore(self, columns : list):
        restored = 0
        if self.deleted is None or self.deleted.empty:
            print("No columns to restore")
            return
        else:
            for col in (set(self.deleted.columns) & set(columns)):
                restored += 1
                _restored = self.deleted[col]

                self.df = pd.concat([self.df, _restored], axis=1, ignore_index=True)
                print(self.df.columns)
                # self.df.append(self.deleted[col])
        print("Restored " + str(restored) + " columns")
            
    
    def applyFilter(self, column, value, maxrows=None, criterion='equal'):
        if maxrows is not None:
            self.df = self.df.head(maxrows)
        if criterion == 'equal':
            self.df = self.df[self.df[column] == value]
        elif criterion == 'contains':
            self.df = self.df[self.df[column].str.contains(value)]

    def colSize(self):
        return len(self.df.columns)
    
    def rowSize(self):
        return len(self.df.index)