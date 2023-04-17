import os
from pathlib import Path
import pandas as pd
import networkx as nx
import numpy as np
from utils import Utilities as utils

class Dataset:
    dataframes = {}
    namespaces = {}
    sources = []
    deleted : pd.DataFrame = None
    df : pd.DataFrame = None
    original_df : pd.DataFrame = None
    base_path = str(Path(os.path.abspath(os.getcwd())))
            
    def addDataset(self, filePath, name=''):
        
        df = original_df = utils.creteDataframe(filePath)
        utils.cleanDataframe(df)
        if df is None:
            print('Error: File not found or not valid')
        else:
            if self.dataframes.get(name) is None:
                self.dataframes[name] = df
            else: 
                self.dataframes[name].append(df)
            print('Added ' + filePath + ' to dataset') 
            
    def createDataset(self, files):
        frames = []
        for file in files:
            mobility_csv = self.base_path + "/data/" + file
            mob_df = pd.read_csv(mobility_csv, sep=";")
            frames.append(mob_df)
        self.df = pd.concat(frames) 
        self.df.drop_duplicates(inplace=True)
        
    def cleanDataframe(self):
        size = self.df.shape[0]
        self.df = utils.cleanDataframe(self.df)
        print("Removed " + str(size - self.df.shape[0]) + " rows")

         
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