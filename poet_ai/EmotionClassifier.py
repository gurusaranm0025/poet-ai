import pandas as pd
import torch

from .config import EmotionClassifierConfig, Config
from .PreProcessing import LabelEncoderC, DatasetC, EmotionDataset, get_dataLoader

from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_scheduler, PreTrainedTokenizerFast, PreTrainedTokenizer
from sklearn.metrics import accuracy_score

class EmotionClassifier:
    def __init__(self, config: Config = EmotionClassifierConfig) -> None:
        self.df: pd.DataFrame = None
        self.tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = None
        self.optimizer = None
        self.lr_scheduler = None

        self.config = config
        self.label_encoder: LabelEncoderC = LabelEncoderC.load(self.config.LABEL_ENCODER_BIN)
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.PRETRAINED_MODEL_NAME, 
            num_labels=len(self.label_encoder.classes_),
        )

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)
    
    def train_DL(self,train_dataloader: DataLoader, epochs: int = 1, lr: float = 5e-5):
        if self.optimizer == None:
            self.optimizer = AdamW(self.model.parameters(), lr=lr)

        num_training_steps = epochs * len(train_dataloader)
        
        if self.lr_scheduler == None:
            self.lr_scheduler = get_scheduler(
                "linear",
                optimizer=self.optimizer,
                num_warmup_steps=0,
                num_training_steps=num_training_steps
            )

        for epoch in range(epochs):
            print(f"\n \tEPOCH ==> {epoch+1}")
            self.model.train()
            self.optimizer.zero_grad()
            total_correct = 0
            total_samples = 0
            # batch: dict
            
            progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{epochs}', leave=True, dynamic_ncols=True)

            for batch in train_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)

                correct = (predictions == batch['labels']).sum().item()
                total_correct += correct
                total_samples += batch['labels'].size(0)
                
                # backpropagation
                loss.backward()
                
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                
                progress_bar.set_postfix({'loss': loss.item(), 'accuracy': f'{((total_correct/total_samples)*100):.4f}', 'total_samples': total_samples})
                progress_bar.update(1)
            
            progress_bar.close()
            acc = total_correct / total_samples
            print(f'Epoch {epoch + 1}/{epochs}, Accuracy: {(acc*100):.4f}')
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
                'loss': loss,
            }, self.config.EC_MODEL_PATH)

    def evaluate_DL(self, val_dataloader: DataLoader):
        self.model.eval()
        total_correct = 0
        total_samples = 0
        progress_bar = tqdm(val_dataloader, desc='Evaluating: ', leave=True, dynamic_ncols=True)
        
        for batch in val_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = self.model(**batch)
            
            loss = outputs.loss
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            
            correct = (predictions == batch['labels']).sum().item()
            total_correct += correct
            total_samples += batch['labels'].size(0)
            
            progress_bar.set_postfix({'loss': loss.item(), 'accuracy':  f'{((total_correct/total_samples)*100):.4f}', 'total_samples': total_samples})
            progress_bar.update(1)
        
        progress_bar.close()
    
    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.PRETRAINED_MODEL_NAME)
    
    def load_model(self):
        checkpoint = torch.load(self.config.EC_MODEL_PATH)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=0
        )
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        print("\n MODEL LOADED.")
        
    def predict(self, input_text: str):
        self.model.eval()
        
        if self.tokenizer == None:
            self.load_tokenizer()
            
        tokens = self.tokenizer(input_text, return_tensors='pt', truncation=True, padding=True, max_length=78)
        input_ids = tokens['input_ids'].to(self.device)
        am = tokens['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=am)
        
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()
        
        pred_cls = self.label_encoder.inverse_transform([prediction])[0]
        
        return pred_cls
    
    def predict_many(self, poems: pd.DataFrame):
        y_pred = poems.apply(self.predict)
        y_pred.columns = ['label']
        y_pred.to_csv("./prediction.csv", index=False)
    
    def evaluate(self, df: pd.DataFrame):
        y_true = df[self.config.TARGET_COLUMN_NAME].tolist()
        y_pred = df[self.config.POEM_COLUMN_NAME].apply(self.predict).tolist()
        
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        
        print(f"\n ACCURACY ==> {(acc*100):.4f}") 

if __name__ == "__main__":
    print("==========================================================")
    emotion_classifier = EmotionClassifier()
    emotion_classifier.load_model()
    
    df = pd.read_csv(EmotionClassifierConfig.UNLABELED_POEMS)
    emotion_classifier.predict_many(df['content'])
    
    # df = pd.read_csv(EmotionClassifierConfig.LABELED_POEMS)
    # emotion_classifier.evaluate(df)

    # dataset = DatasetC()
    # tdl = dataset.laod_get_TrainDL()
    # vdl = dataset.load_get_ValDL()
    # emotion_classifier.train_DL(tdl, epochs=5)
    # emotion_classifier.evaluate_DL(vdl)
    
