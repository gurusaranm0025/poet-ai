import os

class Config():
    DATASET_FOLDER: str = "./datasets"
    
    LABELED_POEMS = os.path.join(DATASET_FOLDER, "labeled_poems.csv")
    COLS_DROP_LP = ['type', 'age', 'pred', 'score']
    EMOTION_COLS_LP = ['anger', 'neutral', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
    
    PERC_EXPERT = os.path.join(DATASET_FOLDER, "PERC-expert.csv")
    EMOTIONS_CSV = os.path.join(DATASET_FOLDER, "emotions.csv")

    MASTER_DATASET = os.path.join(DATASET_FOLDER, "master-dataset.csv")
    
    PREPROCESSED_DATASET = os.path.join(DATASET_FOLDER, "master-dataset-pp.csv")
    
    POEM_COLUMN_NAME = 'poem'
    TARGET_COLUMN_NAME = 'label'
    
    UNLABELED_POEMS = os.path.join(DATASET_FOLDER, "unlabelled_poems.csv")
    
    PRETRAINED_MODEL_NAME: str
    PREPROCESSED_TRAIN_SET: str
    PREPROCESSED_VAL_SET: str
    LABEL_ENCODER_BIN: str
    EC_MODEL_PATH: str

class EmotionClassifierConfig(Config):
    PRETRAINED_MODEL_NAME = "bert-base-uncased"
    PREPROCESSED_TRAIN_SET = "./datasets/PREPROCESSED_TRAIN_SET_EC.csv"
    PREPROCESSED_VAL_SET = "./datasets/PREPROCESSED_VAL_SET_EC.csv"
    LABEL_ENCODER_BIN = "./bin/label_encoder_EC.bin"
    EC_MODEL_PATH = "./bin/EC_model.bin"
