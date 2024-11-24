import numpy as np 
import joblib
import pandas as pd 
import matplotlib.pyplot as plt 

# import tensorflow.keras.utils as ku 

from wordcloud import WordCloud 

# from tensorflow.keras.preprocessing.sequence import pad_sequences 
# from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional 
# from tensorflow.keras.preprocessing.text import Tokenizer 
# from tensorflow.keras.models import Sequential 
# from tensorflow.keras.optimizers import Adam 
# from tensorflow.keras import regularizers
from tqdm.keras import TqdmCallback
from keras import utils as ku
from keras import layers, models, optimizers, regularizers, preprocessing
from keras_preprocessing.text import Tokenizer

from ..config import LSTM_Config

# Reading the text data file 
data = open(LSTM_Config.TEXT_DATASET_PATH, encoding="utf8").read() 

# EDA: Generating WordCloud to visualize 
# the text 
# wordcloud = WordCloud(max_font_size=50, 
# 					max_words=100, 
# 					background_color="black").generate(data) 

# # Plotting the WordCloud 
# plt.figure(figsize=(8, 4)) 
# plt.imshow(wordcloud, interpolation='bilinear') 
# plt.axis("off") 
# plt.savefig("WordCloud.png") 
# plt.show()

# Generating the corpus by 
# splitting the text into lines 
corpus = data.lower().split("\n")
print(corpus[:10])

# Fitting the Tokenizer on the Corpus 
tokenizer = Tokenizer() 
tokenizer.fit_on_texts(corpus) 

# Vocabulary count of the corpus 
total_words = len(tokenizer.word_index) 

print("Total Words:", total_words) 

# Converting the text into embeddings 
input_sequences = [] 
for line in corpus: 
	token_list = tokenizer.texts_to_sequences([line])[0] 

	for i in range(1, len(token_list)): 
		n_gram_sequence = token_list[:i+1] 
		input_sequences.append(n_gram_sequence) 

max_sequence_len = max([len(x) for x in input_sequences]) 
input_sequences = np.array(preprocessing.sequence.pad_sequences(input_sequences, 
										maxlen=max_sequence_len, 
										padding='pre')) 
predictors, label = input_sequences[:, :-1], input_sequences[:, -1] 
label = ku.to_categorical(label, num_classes=total_words+1) 

# Building a Bi-Directional LSTM Model 
model = models.Sequential() 
model.add(layers.Embedding(total_words+1, 100, 
					input_length=max_sequence_len-1)) 
model.add(layers.Bidirectional(layers.LSTM(150, return_sequences=True))) 
model.add(layers.Dropout(0.2)) 
model.add(layers.LSTM(100)) 
model.add(layers.Dense(int(total_words+1/2), activation='relu', 
				kernel_regularizer=regularizers.l2(0.01))) 
model.add(layers.Dense(total_words+1, activation='softmax')) 
model.compile(loss='categorical_crossentropy', 
			optimizer='adam', metrics=['accuracy']) 
print(model.summary()) 

history = model.fit(predictors, label, epochs=170, verbose=1, callbacks=[TqdmCallback()])

model.save(LSTM_Config.MODEL_PATH)
joblib.dump(tokenizer, LSTM_Config.TOKENIZER_PATH)
joblib.dump(max_sequence_len, LSTM_Config.MAX_SEQ_PATH)
joblib.dump(history, LSTM_Config.TRAINING_HISTORY_BIN)

seed_text = "The world"
next_words = 25
ouptut_text = "" 

for _ in range(next_words): 
	token_list = tokenizer.texts_to_sequences([seed_text])[0] 
	token_list = preprocessing.sequence.pad_sequences( 
		[token_list], maxlen=max_sequence_len-1, 
	padding='pre') 
	predicted = np.argmax(model.predict(token_list, 
										verbose=0), axis=-1) 
	output_word = "" 
	
	for word, index in tokenizer.word_index.items(): 
		if index == predicted: 
			output_word = word 
			break
			
	seed_text += " " + output_word 
	
print(seed_text) 
