from typing import List
import pandas as pd
import torch
import ast

from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW, get_scheduler, PreTrainedTokenizerBase, BartTokenizer, BartForConditionalGeneration
from tqdm import tqdm
from datasets import Dataset
from sklearn.model_selection import train_test_split

from .config import GEN_Config, Config

class PoemDataset(Dataset):
    def __init__(self, input_ids, am, labels) -> None:
        self.input_ids: list[str] = input_ids
        self.am = am
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor([ast.literal_eval(x) for x in self.input_ids[idx]]),
            'attention_mask': torch.tensor([ast.literal_eval(x) for x in self.am[idx]]),
            'labels': torch.tensor([ast.literal_eval(x) for x in self.labels[idx]])
        }

class PreProcessor:
    def __init__(self, emotions = None, poems = None, config = GEN_Config) -> None:
        self.config = config
        self.emotions = emotions
        self.poems = poems
        self.tokenizer: PreTrainedTokenizerBase = BartTokenizer.from_pretrained(self.config.PRETRAINED_MODEL_NAME)
    
    def save_dataset(self, df: pd.DataFrame):
        poems = df[self.config.POEM_COLUMN_NAME]
        emotions = df[self.config.TARGET_COLUMN_NAME]
        
        poem_encodings = self.tokenizer(list(poems), truncation=True, padding="max_length", max_length=512)
        emotions_encodings = self.tokenizer([f'generate poem: {emotion}' for emotion in emotions], truncation=True, padding="max_length", max_length=512)
        
        df = pd.DataFrame({})
        df['input_ids'] = emotions_encodings['input_ids']
        df['attention_mask'] = emotions_encodings['attention_mask']
        df['labels'] = poem_encodings['input_ids']
        
        _, val_df = train_test_split(df, test_size=0.1)
        
        self.tokenizer.save_pretrained(self.config.TOKENIZER_PATH)
        val_df.to_csv(self.config.PREPROCESSED_VAL_SET, index=False)
        df.to_csv(self.config.PREPROCESSED_TRAIN_SET, index=False)
    
    def get_dataloaders(self):
        train_df = pd.read_csv(self.config.PREPROCESSED_TRAIN_SET)
        eval_df = pd.read_csv(self.config.PREPROCESSED_VAL_SET)
                
        train_set = PoemDataset(input_ids=train_df['input_ids'], am=train_df['attention_mask'], labels=train_df['labels'])
        eval_set = PoemDataset(input_ids=eval_df['input_ids'].values, am=eval_df['attention_mask'].values, labels=eval_df['labels'].values)
        
        return [DataLoader(train_set, batch_size=1), DataLoader(eval_set, batch_size=1)]

class Poet_AI:
    def __init__(self, config = GEN_Config, train_DL: DataLoader = None, eval_DL: DataLoader = None) -> None:
        self.config = config
        self.model = BartForConditionalGeneration.from_pretrained(config.PRETRAINED_MODEL_NAME)
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5)
        self.lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=0
        )
        
        self.device = torch.device("cude") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)
        
        self.train_DL: DataLoader = train_DL
        self.eval_DL: DataLoader = eval_DL
        self.tokenizer: PreTrainedTokenizerBase = None
    
    def load_tokenizer(self):
        if self.tokenizer == None:
            self.tokenizer = BartTokenizer.from_pretrained(self.config.TOKENIZER_PATH)

    def train_model_DL(self, epochs: int = 1, train_DL: DataLoader = None):
        if train_DL == None and self.train_DL == None:
            raise ValueError("No training dataset found or given for training.")

        if train_DL == None and self.train_DL != None:
            train_DL = self.train_DL
        
        num_train_steps = epochs * len(train_DL)
        self.lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_train_steps
        )
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            total_samples = 0
            
            prog_bar = tqdm(train_DL, desc=f"Epoch: {epoch+1}/{epochs}", leave=True, dynamic_ncols=True)
            
            for batch in train_DL:                        
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                
                total_loss += loss.item()
                total_samples += batch['labels'].size(0)
                prog_bar.set_postfix({'loss': loss.item(), 'total_samples': total_samples})
                prog_bar.update(1)
            
            prog_bar.close()
            avg_loss = total_loss / len(train_DL)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
                'loss': total_loss,
            }, self.config.GEN_MODEL_PATH)
    
    def load_model(self):
        checkpoint = torch.load(self.config.GEN_MODEL_PATH)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=0
        )
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        print("\n ==> MODEL LOADED.")

    
    def generate_poem(self, emotion):
        self.load_tokenizer()
        
        # input_text = f'emotion'
        inputs = self.tokenizer(emotion, max_length=512, return_tensors='pt', truncation=True, padding=True).to(self.device)
        
        outputs = self.model.generate(**inputs, max_length=150, num_beams=5, early_stopping=True)
        
        print(outputs[0])
        print(outputs)
        
        generated_poem = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_poem
    
if __name__ == "__main__":
    # df = pd.read_csv(GEN_Config.MASTER_DATASET)

    # preprocessor = PreProcessor()
    # preprocessor.save_dataset(df)
        
    # train_DL, _ = preprocessor.get_dataloaders()
    poet_ai = Poet_AI()
    # poet_ai.train_model_DL(train_DL=train_DL)
    # poet_ai.load_model()
    poemg = poet_ai.generate_poem("generate a poem that expresses the beauty of life")
    print("\n GENERATED POEM ==> \n", poemg)