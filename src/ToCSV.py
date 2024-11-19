import pandas as pd
import numpy as np

from .config import EmotionClassifierConfig

COLS_DROP = ['type', 'age', 'pred', 'score']
EMOTION_COLUMNS = ['anger', 'neutral', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
POEM_COLUMN_NAME = 'poem'
TARGET_COLUMNS_NAME = 'label'

def fromParqToCSV(dataset_path: str):
    df = pd.read_parquet(dataset_path)
    df.columns = [POEM_COLUMN_NAME, TARGET_COLUMNS_NAME]
    
    df[TARGET_COLUMNS_NAME] = df[TARGET_COLUMNS_NAME].replace(0, 'sad')
    df[TARGET_COLUMNS_NAME] = df[TARGET_COLUMNS_NAME].replace(1, 'joy')
    df[TARGET_COLUMNS_NAME] = df[TARGET_COLUMNS_NAME].replace(2, 'love')
    df[TARGET_COLUMNS_NAME] = df[TARGET_COLUMNS_NAME].replace(3, 'anger')
    df[TARGET_COLUMNS_NAME] = df[TARGET_COLUMNS_NAME].replace(4, 'fear')
    df[TARGET_COLUMNS_NAME] = df[TARGET_COLUMNS_NAME].replace(5, 'surprise')
    
    print(df[TARGET_COLUMNS_NAME].unique())
    df.to_csv("./datasets/emotions.csv", index=False)
    print(df.head())

def toCSV(dataset_path: str):
    df = pd.read_excel(dataset_path)
    print(df['Emotion'].unique())
    df.to_csv('./datasets/PERC-expert.csv')

def master_df(dataset1: str, dataset2: str, dataset3: str):
    df1 = pd.read_csv(dataset1)
    df1.drop(columns=COLS_DROP+EMOTION_COLUMNS, inplace=True)
    df1.columns = [POEM_COLUMN_NAME, TARGET_COLUMNS_NAME]
    print(df1.shape)
    
    df2 = pd.read_csv(dataset2)
    df2.columns = ['index', POEM_COLUMN_NAME, TARGET_COLUMNS_NAME]
    df2.drop(columns=['index'], inplace=True)
    print(df2.shape)
    
    df3 = pd.read_csv("./datasets/emotions.csv")
    print(df3.shape)
    
    df = pd.concat([df1, df2, df3], ignore_index=True)
    df[TARGET_COLUMNS_NAME] = np.where(df[TARGET_COLUMNS_NAME]=='sadness', 'sad', df[TARGET_COLUMNS_NAME])
    print(df[TARGET_COLUMNS_NAME].unique())
    print(df.shape)
    df.to_csv('./datasets/master-dataset.csv', index=False)

def LP_csv():
    df1 = pd.read_csv(EmotionClassifierConfig.LABELED_POEMS)
    df1.drop(columns=COLS_DROP+EMOTION_COLUMNS, inplace=True)
    df1.columns = [POEM_COLUMN_NAME, TARGET_COLUMNS_NAME]
    df1[TARGET_COLUMNS_NAME] = np.where(df1[TARGET_COLUMNS_NAME]=='sadness', 'sad', df1[TARGET_COLUMNS_NAME])
    df1.to_csv(EmotionClassifierConfig.LABELED_POEMS, index=False)

    
if __name__ == "__main__":
    # toCSV('./datasets/PERC-expert.xlsx')
    # fromParqToCSV("./datasets/emotion-hf/emotions.parquet")
    # master_df('./datasets/labeled_poems.csv', './datasets/PERC-expert.csv', './datasets/emotions.csv')
    LP_csv()