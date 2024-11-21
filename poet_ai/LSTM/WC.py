import matplotlib.pyplot as plt
from wordcloud import WordCloud 

from ..config import LSTM_Config

data = open(LSTM_Config.DATASET_PATH, encoding="utf8").read()

wordcloud = WordCloud(max_font_size=50, max_words=200, background_color="black").generate(data)

plt.figure(figsize=(8, 4))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig("wordcloud.png")
plt.show()