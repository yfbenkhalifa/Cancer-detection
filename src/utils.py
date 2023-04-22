import csv
import pandas

class Utilities:
    def __init__(self):
        pass
    
    @staticmethod
    def createDataframe(filepath : str , separator = ';') -> pandas.DataFrame:
        format = filepath.split('.')[-1]
        if format == 'csv':
            return pandas.DataFrame(pandas.read_csv(filepath, sep = separator, header = 0))
        elif format == 'xlsx':
            return pandas.read_excel(filepath, engine="openpyxl")
        elif format == 'tsv':
            return pandas.DataFrame(pandas.read_csv(filepath, sep='\t', header = 0))
        else:
            return None
            
    @staticmethod
    def loadDatasets(dataDir, datasets):
        dataframes = []
        for dataset in datasets:
            for datasrouce in datasets:
                csvFile = dataDir + datasrouce[0]
                df = Utilities.creteDataframe(csvFile)
                if df is None:
                    print('Error: File not found or not valid')
                else:
                    dataframes.__add__([df, datasrouce[1]])
        return dataframes
    
    @staticmethod
    def cleanDataframe(dataframe:pandas.DataFrame, ):
        dataframe.dropna(axis=0, how='any', inplace=True)
        dataframe.dropna(axis=1, how='all', inplace=True)
        return dataframe
    
    @staticmethod
    def splitDataFrame(dataframe:pandas.DataFrame, columns : list):
        for column in columns:
            column = column.lower()
        return dataframe[columns]