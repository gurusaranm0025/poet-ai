import os

class Config():
    DATASET_FOLDER: str = "./datasets"
    
    LABELED_POEMS = os.path.join(DATASET_FOLDER, "labeled_poems.csv")
    COLS_DROP_LP = ['type', 'age', 'pred', 'score']
    EMOTION_COLS_LP = ['anger', 'neutral', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
    
    PERC_EXPERT = os.path.join(DATASET_FOLDER, "PERC-expert.csv")
    EMOTIONS_CSV = os.path.join(DATASET_FOLDER, "emotions.csv")

    
    PREPROCESSED_DATASET = os.path.join(DATASET_FOLDER, "master-dataset-pp.csv")
    
    POEM_COLUMN_NAME = 'poem'
    TARGET_COLUMN_NAME = 'label'
    
    UNLABELED_POEMS = os.path.join(DATASET_FOLDER, "unlabeled_poems.csv")
    COLS_DROP_ULP = ['author', 'age', 'poem name']
    
    LABEL_ENCODER_BIN = "./bin/label_encoder_EC.bin"
    
    MASTER_DATASET: str
    PRETRAINED_MODEL_NAME: str
    PREPROCESSED_TRAIN_SET: str
    PREPROCESSED_VAL_SET: str
    LABEL_ENCODER_BIN: str
    EC_MODEL_PATH: str

class EmotionClassifierConfig(Config):
    MASTER_DATASET = os.path.join(Config.DATASET_FOLDER, "master-dataset.csv")
    PRETRAINED_MODEL_NAME = "bert-base-uncased"
    PREPROCESSED_TRAIN_SET = "./datasets/PREPROCESSED_TRAIN_SET_EC.csv"
    PREPROCESSED_VAL_SET = "./datasets/PREPROCESSED_VAL_SET_EC.csv"
    EC_MODEL_PATH = "./bin/EC_model.bin"

class GEN_Config(Config):
    MASTER_DATASET = os.path.join(Config.DATASET_FOLDER, "master-gen-dataset.csv")
    
    # PRETRAINED_MODEL_NAME = "t5-small"
    PRETRAINED_MODEL_NAME = "facebook/bart-large"
    
    TARGET_COLUMN_NAME = "labels"
    PREPROCESSED_TRAIN_SET = os.path.join(Config.DATASET_FOLDER, "GEN_PREPROCESSED_TRAIN.csv")
    PREPROCESSED_VAL_SET = os.path.join(Config.DATASET_FOLDER, "GEN_PREPROCESSED_VAL.csv")
    
    # GEN_MODEL_PATH: str = "./bin/GEN_MODEL.bin"
    # TOKENIZER_PATH: str = "./bin/GEN_TOKENIZER.bin"
    GEN_MODEL_PATH: str = "./bin/GEN_BART.bin"
    TOKENIZER_PATH: str = "./bin/GEN_BART_TOKENIZER.bin"

class LSTM_Config:
    BIN_FOLDER = "bin"
    DATASET_FOLDER = "./datasets/poems"
    POEMS_TEXT_FILE = "poems.txt"
    TEXT_DATASET_PATH = os.path.join(DATASET_FOLDER, POEMS_TEXT_FILE)
    
    POEM_CSV_FILE = "poems.csv"
    POEM_COL_NAME = "poem"
    CSV_DATASET_PATH = os.path.join(DATASET_FOLDER, POEM_CSV_FILE)
    
    DATASET_BATCH_SIZE = 32
    
    MODEL_FILE_NAME = "LSTM_POET.h5"
    MODEL_PATH = os.path.join(BIN_FOLDER, MODEL_FILE_NAME)
    
    TOKENIZER_FILE_NAME = "LSTM_POET_TOKENIZER.bin"
    TOKENIZER_PATH = os.path.join(BIN_FOLDER, TOKENIZER_FILE_NAME)
    
    MAX_SEQ_FILE_NAME = "MAX_SEQ.bin"
    MAX_SEQ_PATH = os.path.join(BIN_FOLDER, MAX_SEQ_FILE_NAME)