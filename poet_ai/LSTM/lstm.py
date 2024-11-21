import warnings
import joblib
import logging
import numpy as np
import pandas as pd
import time
import os

from types import NoneType
from tensorflow import data
from tqdm.keras import TqdmCallback
from keras import utils as ku
from keras import layers, models, regularizers, preprocessing
from keras_preprocessing.text import Tokenizer

from ..config import LSTM_Config

warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

class LSTM_Poet:
    def __init__(self, config = LSTM_Config) -> None:
        self._config = config
        self._tokenizer = Tokenizer()
        self._input_sequences: np.ndarray | list = []
        self._corpus: list = []
        
        self.total_words_: str = None
        self.max_sequence_len_: int = None
        self._history = None
        self.dataset: data.Dataset = None
        self._predictors: np.ndarray = None
        self._label: np.ndarray = None
        self._model: models.Sequential = None
    
    def process_corpus(self):
        self._tokenizer.fit_on_texts(self._corpus)
        self.total_words_ = len(self._tokenizer.word_index)
        
        for line in self._corpus:
            tokens = self._tokenizer.texts_to_sequences([line])[0]
            
            for i in range(len(tokens)):
                self._input_sequences.append(tokens[:i+1])

        self.max_sequence_len_ = max([len(x) for x in self._input_sequences])
        self._input_sequences = np.array(preprocessing.sequence.pad_sequences(self._input_sequences, maxlen=self.max_sequence_len_, padding='pre'))
        
        self._predictors, self._label = self._input_sequences[:, :-1], self._input_sequences[:, -1]
        
        self._label = ku.to_categorical(self._label, num_classes=self.total_words_+1)
        
        # self._get_dataset()
    
    def _get_dataset(self, batch_size: int = None):
        if batch_size == None:
            batch_size = self._config.DATASET_BATCH_SIZE
        
        self.dataset = data.Dataset.from_tensor_slices((self._predictors, self._label))
        self.dataset = self.dataset.shuffle(buffer_size=10000)
        self.dataset = self.dataset(batch_size)
        self.dataset = self.dataset.prefetch(buffer_size=data.experimental.AUTOTUNE)
        
        self.shrink()
        self._predictors = None
        self._label = None
    
    def load_corpus(self, dataset_txt_filepath: str = None):
        if dataset_txt_filepath == None:
            dataset_txt_filepath = self._config.TEXT_DATASET_PATH
            
        corpus = open(dataset_txt_filepath, encoding='utf8').read()
        self._corpus += corpus.lower().split("\n")
        
    
    def load_corpus_from_df(self, df: pd.DataFrame = None, poem_col: str = None):
        if df == None:
            df = pd.read_csv(self._config.CSV_DATASET_PATH)
        
        if poem_col == None:
            poem_col = self._config.POEM_COL_NAME
                        
        for _, row in df.iterrows():
            poem: str = row[poem_col]
            
            for line in poem.lower().split("\n"):
                stripped_line = line.strip()
                if not stripped_line == '':
                    self._corpus.append(stripped_line)
    
    def corpus_summary(self):
        print("\n\t CORPUS SUMMARY => ")
        print(f'\t\tNUMBER OF LINES ==> {len(self._corpus)}')
        
        print("\n\t\t FEW SAMPLES ==>")
        print(f"\n\t\t{self._corpus[:10]}")
                
    def _init_model(self):
        self._model = models.Sequential()
        
        self._model.add(layers.Embedding(self.total_words_+1, 100, input_length=self.max_sequence_len_-1))
        
        self._model.add(layers.Bidirectional(layers.LSTM(150, return_sequences=True)))
        
        self._model.add(layers.Dropout(0.2))
        
        self._model.add(layers.LSTM(100))
        
        self._model.add(layers.Dense(int(self.total_words_+1/2), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        
        self._model.add(layers.Dense(self.total_words_+1, activation='softmax'))
        
        self._model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print("\n\t ==> MODEL INTIATED")
    
    def model_summary(self):
        print(self._model.summary())
    
    def fit(self, epochs: int = 1, verbose: int = 0, save_poet: bool = False):
        if (type(self._label == NoneType)) and (not len(self._corpus) > 0):
            raise ValueError("No dataset is provided, for training.")

        if type(self._label) == NoneType:
            self.process_corpus()
            
        if self._model == None:
            self._init_model()
        else:
            warnings.warn("Model is already trained on a dataset. Training again will affect already trained model.")
            time.sleep(2)
            
        self._history = self._model.fit(self._predictors, self._label, epochs=epochs, verbose=verbose, callbacks=[TqdmCallback()])
        
        if save_poet:
            self.save_lstm_poet()
    
    def get_history(self):
        if self._history == None:
            raise ValueError("Train the model first, then you can see the history.")
            
        print(f"\n {self._history}")
    
    def generate_poem(self, seed_text: str, next_words: int = 25, verbose: int = 0):
        out = seed_text
        for _ in range(next_words):
            if self._model == None:
                raise ValueError("Train the model first, then you can generate some poems.")
                
            token_list = self._tokenizer.texts_to_sequences([seed_text])[0]
            token_list = preprocessing.sequence.pad_sequences([token_list], maxlen=self.max_sequence_len_-1, padding='pre')
            
            output = np.argmax(self._model.predict(token_list, verbose=verbose), axis=-1)
            
            output_word = ""
            
            for word, index in self._tokenizer.word_index.items():
                if index == output:
                    output = word
                    break
            out += " " + output_word
        
        return out
    
    def save_model(self, filepath: str = None):
        if filepath == None:
            filepath = self._config.MODEL_PATH
        
        self._model.save(filepath)
    
    def save_tokenizer(self, filepath: str = None):
        if filepath == None:
            filepath = self._config.TOKENIZER_PATH
        
        joblib.dump(self._tokenizer, filepath)
        joblib.dump(self.max_sequence_len_, self._config.MAX_SEQ_PATH)
    
    def save_lstm_poet(self):
        self.save_model()
        self.save_tokenizer()
    
    def load_model(self, filepath: str = None):
        if filepath == None:
            filepath = self._config.MODEL_PATH
        
        self._model = models.load_model(filepath)
    
    def load_tokenizer(self, filepath: str = None):
        if filepath == None:
            filepath = self._config.TOKENIZER_PATH
        
        self._tokenizer = joblib.load(filepath)
        self.max_sequence_len_ = joblib.load(self._config.MAX_SEQ_PATH)
        
    
    def load_lstm_poet(self):
        self.load_model()
        self.load_tokenizer()
    
    def shrink(self, full_clean: bool = False):
        if not full_clean:
            if not self._model ==  None:
                if len(self._input_sequences) > 0:
                    print("\n CORPSU IS EMPTIED.")
                    self._corpus = None
                
                if len(self._label) > 0:
                    print("INPUT SEQUENCE IS EMPTIED.")
                    self._input_sequences = None
        else:            
            self._corpus = None
            self._input_sequences = None
            self._predictors = None
            self._label = None
            self._history = None
            print("\n ===> EVERY DATA EXCEPT tokenizer and model HAVE BEEN EMPTIED.")
                
    
if __name__ == "__main__":
    poet = LSTM_Poet()
    poet.load_corpus()
    # poet.process_corpus()
    # poet.load_corpus_from_df()
    poet.corpus_summary()
    poet.fit(epochs=300, save_poet=True)
    # poet.load_lstm_poet()
    print(poet.generate_poem("The world"))    
    