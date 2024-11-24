import warnings
import joblib
import logging
import numpy as np
import pandas as pd
import time
import os

from types import NoneType
from tqdm import tqdm
from tensorflow import data
from tqdm.keras import TqdmCallback
from keras import utils as ku
from keras import layers, models, regularizers, preprocessing, optimizers, initializers
from keras_preprocessing.text import Tokenizer
from gensim.models import Word2Vec

from ..config import LSTM_Config

warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

class LSTM_Poet:
    def __init__(self, config = LSTM_Config, embedding_dim: int = 100) -> None:
        self._config = config
        self._tokenizer = Tokenizer()
        self._input_sequences: np.ndarray | list = []
        self._corpus: list = []
        self.embedding_dim: int = embedding_dim

        self.total_words_: str = None
        self.max_sequence_len_: int = None
        self._history = None
        self.dataset: data.Dataset = None
        self._predictors: np.ndarray = None
        self._label: np.ndarray = None
        self._model: models.Sequential = None
        self.embedding_matrix: np.ndarray = None
    
    def fit_tokenizer(self):
        corpus: str | list[str]
        
        corpus = open(self._config.TEXT_DATASET_PATH, encoding='utf8').read()
        corpus = corpus.lower().split("\n")

        df = pd.read_csv(self._config.CSV_DATASET_PATH)
        for _, row in df.iterrows():
            poem: str = row[self._config.POEM_COL_NAME]
            
            for line in poem.lower().split("\n"):
                stripped_line = line.strip()
                if not stripped_line == '':
                    corpus.append(stripped_line)

        self._tokenizer.fit_on_texts(corpus)
        self.total_words_ = len(self._tokenizer.word_index)
        print("\n \t==> TOKENIZER FITTED.")

    
    def process_corpus(self):
        self._tokenizer = Tokenizer()
        self._tokenizer.fit_on_texts(self._corpus)
        self.total_words_ = len(self._tokenizer.word_index) + 1
        self._input_sequences = []

        for line in self._corpus:
            tokens = self._tokenizer.texts_to_sequences([line])[0]

            for i in range(len(tokens)):
                self._input_sequences.append(tokens[:i+1])

        self.max_sequence_len_ = max([len(x) for x in self._input_sequences])
        self._input_sequences = np.array(preprocessing.sequence.pad_sequences(self._input_sequences, maxlen=self.max_sequence_len_, padding='pre'))

        self._predictors, self._label = self._input_sequences[:, :-1], self._input_sequences[:, -1]

        self._label = ku.to_categorical(self._label, num_classes=self.total_words_)
        print("\n\t ==> CORPUS PROCESSED.")
        # self._get_dataset()
    
    def word2vec_embeddings(self):
        self._word2vec_model = Word2Vec(self._corpus, vector_size=100, window=5, min_count=1, workers=4)
        
        self.embedding_matrix = np.zeros((self.total_words_, self.embedding_dim))
        for word, i in self._tokenizer.word_index.items():
            if word in self._word2vec_model.wv:
                self.embedding_matrix[i] = self._word2vec_model.wv[word]

        print("embed matrix")
        print(self.embedding_matrix)

    # def _get_dataset(self, batch_size: int = None):
    #     if batch_size == None:
    #         batch_size = self._config.DATASET_BATCH_SIZE
        
    #     self.dataset = data.Dataset.from_tensor_slices((self._predictors, self._label))
    #     self.dataset = self.dataset.shuffle(buffer_size=10000)
    #     self.dataset = self.dataset(batch_size)
    #     self.dataset = self.dataset.prefetch(buffer_size=data.experimental.AUTOTUNE)
        
    #     self.shrink()
    #     self._predictors = None
    
    def load_corpus(self, dataset_txt_filepath: str = None):
        if dataset_txt_filepath == None:
            dataset_txt_filepath = self._config.TEXT_DATASET_PATH
            
        corpus = open(dataset_txt_filepath, encoding='utf8').read()
        self._corpus = corpus.lower().split("\n")
        
    
    def load_corpus_from_df(self, df: pd.DataFrame = pd.DataFrame({}), poem_col: str = None):
        if df.empty:
            df = pd.read_csv(self._config.CSV_DATASET_PATH)
        
        if poem_col == None:
            poem_col = self._config.POEM_COL_NAME
                        
        for _, row in df.iterrows():
            poem: str = row[poem_col]
            
            for line in poem.lower().split("\n"):
                stripped_line = line.strip()
                if not stripped_line == '':
                    self._corpus.append(stripped_line)
        print("\n\t ==> CORPUS READY.")
    
    def corpus_summary(self):
        print("\n\t CORPUS SUMMARY => ")
        print(f'\t\tNUMBER OF LINES ==> {len(self._corpus)}')
        
        print("\n\t\t FEW SAMPLES ==>")
        print(f"\n\t\t{self._corpus[:10]}")
                
    def _init_model(self):
        # self._model = models.Sequential()

        self.word2vec_embeddings()

        input_seq = layers.Input(shape=(self.max_sequence_len_-1,))

        embed = layers.Embedding(self.total_words_, self.embedding_dim, embeddings_initializer=initializers.Constant(self.embedding_matrix), input_length=self.max_sequence_len_-1, trainable=False)(input_seq)
        
        lstm_out = layers.Bidirectional(layers.LSTM(200, return_sequences=True))(embed)
        lstm_out = layers.Dropout(0.2)(lstm_out)
        # self._model.add(layers.Dropout(0.2))
        
        # attention = layers.Attention()([lstm_out, lstm_out])
        # atten_out = layers.Concatenate()([lstm_out, attention])
        
        # self._model.add(layers.Bidirectional(layers.LSTM(150, return_sequences=True)))
        # self._model.add(layers.Dropout(0.1))
        
        lstm_out2 = layers.LSTM(150)(lstm_out)
        lstm_out2 = layers.Dropout(0.2)(lstm_out2)
        
        # self._model.add(layers.Dense(int(self.total_words_+1/2), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        
        # self._model.add(layers.Dense(self.total_words_+1, activation='softmax'))
        
        output = layers.Dense(self.total_words_, activation='softmax')(lstm_out2)
        
        self._model = models.Model(inputs=input_seq, outputs=output)
        
        self._model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print("\n\t ==> MODEL INTIATED")
    
    def model_summary(self):
        if self._model == None:
            self._init_model()
            
        print(self._model.summary())
    
    def fit(self, epochs: int = 1, verbose: int = 0, save_poet: bool = False, callbacks: list = [TqdmCallback()]):
        if (type(self._label) == NoneType) and (not len(self._corpus) > 0):
            raise ValueError("No dataset is provided, for training.")

        if type(self._label) == NoneType:
            self.process_corpus()

        if self._model == None:
            self._init_model()
        else:
            warnings.warn("Model is already trained on a dataset. Training again will affect already trained model.")
            time.sleep(2)
        
        # print(self._predictors)
        # print(self._label)
        # print(epochs)
        # print(verbose)
        # print(callbacks)
        self._history = self._model.fit(self._predictors, self._label, epochs=epochs, verbose=verbose, callbacks=callbacks)
        
        if save_poet:
            self.save_lstm_poet()
    
    def get_history(self):
        if self._history == None:
            raise ValueError("Train the model first, then you can see the history.")
            
        print(f"\n {self._history}")
    
    def generate_poem(self, seed_text: str, next_words: int = 25, verbose: int = 0):
        
        if self._model == None:
            raise ValueError("Train the model first, then you can generate some poems.")
        for _ in range(next_words):
                
            token_list = self._tokenizer.texts_to_sequences([seed_text])[0]
            token_list = preprocessing.sequence.pad_sequences([token_list], maxlen=self.max_sequence_len_-1, padding='pre')
            
            pred = np.argmax(self._model.predict(token_list, verbose=verbose), axis=-1)
            
            output_word = ""
            
            for word, index in self._tokenizer.word_index.items():
                if index == pred:
                    output_word = word
                    break
            print(pred)
            print(output_word)
            seed_text += " " + output_word
            print(seed_text)
            print("--------------")
        return seed_text
    
    def save_model(self, filepath: str = None):
        if filepath == None:
            filepath = self._config.MODEL_PATH
        
        self._model.save(filepath)
        joblib.dump(self._history, self._config.TRAINING_HISTORY_BIN)
    
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
        self._history = joblib.load(self._config.TRAINING_HISTORY_BIN)
    
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
    
    def custom_train(self, epochs: int = 300):
        chunks_paths = self._config.get_poem_csv_chunks_paths()
        
        
        for epoch in range(epochs):
            epoch_prog_bar = tqdm(chunks_paths, desc=f'Epoch {epoch+1}/{epochs}', leave=True, dynamic_ncols=True)
            
            for i, path in enumerate(chunks_paths):
                
                df = pd.read_csv(path)
                
                self.load_corpus_from_df(df=df)
                self.corpus_summary()
                self.process_corpus()
                self.fit(verbose=0, epochs=1, save_poet=True)
                
                acc = self._history['accuracy'][-1]
                epoch_prog_bar.set_postfix({'acuracy': acc, 'samples_completed': i+1, 'total_samples': len(chunks_paths)})
                epoch_prog_bar.update(1)
                    
            epoch_prog_bar.close()
    
if __name__ == "__main__":
    poet = LSTM_Poet()
    # poet.fit_tokenizer()
    # poet.load_corpus()
    # poet.process_corpus()
    # poet.corpus_summary()
    # poet.model_summary()
    # poet.fit(epochs=170, save_poet=True)
    poet.load_lstm_poet()
    # poet.custom_train()
    print(poet.generate_poem("The world", next_words=50))    
    