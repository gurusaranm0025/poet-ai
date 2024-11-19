import joblib
import torch
import ast

import pandas as pd

from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer, AutoTokenizer

from .config import EmotionClassifierConfig

class LabelEncoderC(LabelEncoder):
    def save(self, fileName: str):
        joblib.dump(self, fileName)
    
    @classmethod
    def load(self, fileName: str):
        return joblib.load(fileName)

class EmotionDatasetPP(Dataset):
    def __init__(self, encodings, labels) -> None:
        self.encodings = encodings
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.encodings['input_ids'][idx], self.encodings['attention_mask'][idx],self.labels[idx]

class EmotionDataset(Dataset):
    def __init__(self, encodings, am, labels, isDataset: bool = True, isTesting: bool =True) -> None:
        self.encodings = encodings
        self.am = am
        self.labels = labels
        self.isDataset = isDataset
        self.isTesting = isTesting
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if self.isDataset:
            return {
                'input_ids': torch.tensor(ast.literal_eval(self.encodings[idx])),
                'attention_mask': torch.tensor(ast.literal_eval(self.am[idx])),
                'labels': torch.tensor(self.labels[idx], dtype=torch.long)
            }
        else:
            return {
                'input_ids': torch.tensor(self.encodings[idx]),
                'attention_mask': torch.tensor(self.am[idx]),
                'labels': torch.tensor(self.labels[idx], dtype=torch.long)
            }
            

class DatasetC():
    def __init__(self, config = EmotionClassifierConfig) -> None:
        self.df: pd.DataFrame
        self.train_data: EmotionDatasetPP
        self.val_data: EmotionDatasetPP
        
        self.config = config
        self.label_encoder = LabelEncoderC()
        
        self.tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(self.config.PRETRAINED_MODEL_NAME)
    
    def prepare_dataset(self):
        """
            Prepares the dataset for trianing the model.
        """
        print("\n ==> PREPARING DATASET")
        
        # loading and dropping unneccesary columns in the dataset
        self.df = pd.read_csv(self.config.MASTER_DATASET)
        self.df['label_ids'] = self.label_encoder.fit_transform(self.df['label'])
        self.df.drop(columns=['label'], inplace=True)
        
        train_df, val_df = train_test_split(self.df, stratify=self.df['label_ids'],test_size=0.2, random_state=42)
        
        # encoding and tokenising the poems
        train_encodings = self.tokenizer(list(self.df[self.config.POEM_COLUMN_NAME]), truncation=True, padding=True, max_length=78)
        val_encodings = self.tokenizer(list(val_df[self.config.POEM_COLUMN_NAME]), truncation=True, padding=True, max_length=78)
        
        # separating the target label
        train_labels = self.df['label_ids'].values
        val_labels = val_df['label_ids'].values
        
        print(f"Train labels shape: {self.df.shape}")
        print(f"Validation labels shape: {val_labels.shape}")

        # creating valid datsets for training
        self.train_data = EmotionDatasetPP(train_encodings, train_labels)
        self.val_data = EmotionDatasetPP(val_encodings, val_labels)
        
        self.train_data_loader = DataLoader(self.train_data, batch_size=64, shuffle=True)
        self.val_data_loader = DataLoader(self.val_data, batch_size=16)
        print("\n ==> DATASET PREPARED.")
    
    def save_LE(self):
        self.label_encoder.save(self.config.LABEL_ENCODER_BIN)
    
    def shrink(self, df: bool = True, tokenizer: bool = True, label_encoder: bool = True, train_data: bool = False, val_data: bool = False):
        print("\n Shrinking...")
        if df:
            self.df = None
        if tokenizer:
            self.tokenizer = None
        if label_encoder:
            self.label_encoder = None
        if train_data:
            self.train_data = None
        if val_data:
            self.val_data = None
        self.tokenizer = None
        self.train_data_loader = None
        self.val_data_loader = None
    
    def save_preprocessed_data(self):
        """
            saves the preprocessed dataset as a csv file.
        """
        print("\n Saving")
        if (self.train_data == None) or (self.val_data == None):
            raise ValueError("Can't save a empty data. Load and prepare the dataset.")
        dataset = []
        for encoding, am, label in self.train_data:
            dataset.append([encoding, am, label])
        
        df = pd.DataFrame(dataset, columns=['encoding', 'am', 'label'])
        df.to_csv(self.config.PREPROCESSED_TRAIN_SET)
        df = None
        self.train_data = None
        
        dataset = []
        for encoding, am, label in self.val_data:
            dataset.append([encoding, am, label])
        df = pd.DataFrame(dataset, columns=['encoding', 'am', 'label'])
        df.to_csv(self.config.PREPROCESSED_VAL_SET, index=False)
    
    def laod_get_TrainDL(self) -> DataLoader:
        print("\n Loading Train DL...")
        df = pd.read_csv(self.config.PREPROCESSED_TRAIN_SET)
        train_data = EmotionDataset(encodings=df['encoding'], am=df['am'], labels=df['label'])
        return DataLoader(train_data, batch_size=64, shuffle=True, num_workers=8)
    
    def load_get_ValDL(self) -> DataLoader:
        print("\n Loading Val DL...")
        df = pd.read_csv(self.config.PREPROCESSED_VAL_SET)
        val_data = EmotionDataset(encodings=df['encoding'], am=df['am'], labels=df['label'])
        return DataLoader(val_data, batch_size=124, num_workers=8)

def get_dataLoader(dataset: EmotionDataset) -> DataLoader:
    return DataLoader(dataset, batch_size=124, num_workers=8)

if __name__ == "__main__":
    dataset = DatasetC()
    dataset.prepare_dataset()
    dataset.save_LE()
    dataset.shrink()
    dataset.save_preprocessed_data()