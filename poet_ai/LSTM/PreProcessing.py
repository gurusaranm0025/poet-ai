import pandas as pd

from ..config import LSTM_Config

def save_csv_chunks(filepath: str = LSTM_Config.CSV_DATASET_PATH, chunk_size: int = LSTM_Config.CHUNK_SIZE):
    chunks: list[pd.DataFrame] = pd.read_csv(filepath, chunksize=chunk_size)
    chunks_paths = LSTM_Config.get_poem_csv_chunks_paths()
    
    for i, chunk in enumerate(chunks):
        chunk.to_csv(chunks_paths[i], index=False)
        print(f"\n ==> SAVED A CHUNK FILE TO '{chunks_paths[i]}'")

if __name__ == "__main__":
    save_csv_chunks()