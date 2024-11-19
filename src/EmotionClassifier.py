import pandas as pd
import torch

from .config import EmotionClassifierConfig, Config
from .preprocessing import LabelEncoderC, DatasetC, EmotionDataset, get_dataLoader

from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_scheduler, PreTrainedTokenizerFast, PreTrainedTokenizer

class EmotionClassifier:
    def __init__(self, config: Config = EmotionClassifierConfig) -> None:
        self.df: pd.DataFrame = None
        self.tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = None

        self.config = config
        self.label_encoder: LabelEncoderC = LabelEncoderC.load(self.config.LABEL_ENCODER_BIN)
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.PRETRAINED_MODEL_NAME, 
            num_labels=len(self.label_encoder.classes_),
        )
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5)
        self.lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=0
        )

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)
    
    def train_DL(self,train_dataloader: DataLoader, epochs: int = 1, lr: float = 5e-5):
        self.optimizer = AdamW(self.model.parameters(), lr=lr)

        num_training_steps = epochs * len(train_dataloader)
        
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )
                
        accumulation_steps = 4

        for epoch in range(epochs):
            print(f"\n \tEPOCH ==> {epoch+1}")
            self.model.train()
            self.optimizer.zero_grad()
            total_correct = 0
            total_samples = 0
            # batch: dict
            
            progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{epochs}', leave=True, dynamic_ncols=True)

            for step, batch in enumerate(train_dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss = loss / accumulation_steps
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)

                correct = (predictions == batch['labels']).sum().item()
                total_correct += correct
                total_samples += batch['labels'].size(0)
                
                # backpropagation
                loss.backward()
                
                if (step + 1) % accumulation_steps == 0:
                    self.optimizer.step()
                    lr_scheduler.step()
                    self.optimizer.zero_grad()
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                        'loss': loss,
                    }, self.config.EC_MODEL_PATH)

                
                progress_bar.set_postfix({'loss': loss.item(), 'accuracy': f'{((total_correct/total_samples)*100):.4f}', 'total_samples': total_samples})
            
            acc = total_correct / total_samples
            print(f'Epoch {epoch + 1}/{epochs}, Accuracy: {(acc*100):.4f}')
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'loss': loss,
            }, self.config.EC_MODEL_PATH)

    def evaluate_DL(self, val_dataloader: DataLoader):
        self.model.eval()
        total_correct = 0
        total_samples = 0
        progress_bar = tqdm(val_dataloader, desc='Evaluating: ', leave=True, dynamic_ncols=True)
        
        for batch in val_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            print(batch)
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
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        print("\n MODEL LOADED.")
    
    
        
        

if __name__ == "__main__":
    print("==========================================================")
    emotion_classifier = EmotionClassifier()
    emotion_classifier.load_model()
    # dataset = DatasetC()
    # tdl = dataset.laod_get_TrainDL()
    # vdl = dataset.load_get_ValDL()
    # emotion_classifier.train(tdl)
    # emotion_classifier.evaluate_DL(vdl)
    
    df = pd.read_csv(EmotionClassifierConfig.LABELED_POEMS)
    emotion_classifier.evaluate(df)