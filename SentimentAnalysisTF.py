import tensorflow as tf
from nltk.stem.porter import PorterStemmer
import random
import numpy as np
import curses
import tflearn
import json
import nltk
import unicodedata
import sys


stemmer = PorterStemmer()
data = None

with open('F:\\Sentiments.json') as json_data:
    data = json.load(json_data)
    print(data)

Key_categories = list(data.keys())  # collecting the Keys from the key-value pair JSON
words = []
document = []

for each_category in data.keys():
    for each_sentence in data[each_category]:
        wordToken = nltk.word_tokenize(each_sentence)
        print(wordToken)
        words.extend(wordToken)
        document.append((wordToken, each_category))
words = [stemmer.stem(wordToken.lower()) for wordToken in words]
words = sorted(list(set(words)))

print(words)
print(document)

trainingData = []
outputData = []
output_empty = [0] * len(Key_categories)

for doc in document:
    BagOfWords = []
    token_words = doc[0]
    token_words = [stemmer.stem(word.lower()) for word in token_words]
    for w in words:
        BagOfWords.append(1) if w in token_words else BagOfWords.append(0)

    output_row = list(output_empty)
    output_row[Key_categories.index(doc[1])] = 1
    trainingData.append([BagOfWords, output_row])

random.shuffle(trainingData)
training = np.array(trainingData)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

tf.reset_default_graph()
##Conneting the Net
TFnet = tflearn.input_data(shape=[None, len(train_x[0])])
TFnet = tflearn.fully_connected(TFnet, 12)
TFnet = tflearn.fully_connected(TFnet, 12)
TFnet = tflearn.fully_connected(TFnet, len(train_y[0]), activation='softmax')
TFnet = tflearn.regression(TFnet)

model = tflearn.DNN(TFnet)
model.fit(train_x, train_y, n_epoch=1200, batch_size=8, show_metric=True)
model.save('SentimentTensor.tf')


def TensorOut(sentence):
    global words
    sentence_words = nltk.word_tokenize(sentence)

    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]

    bag_of_words = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bow[i] = 1

    return (np.array(bag_of_words))


sentiment_1 = "It was a pathetic movie"
sentiment_2 = "I hated the latest thor"
sentiment_3 = "justice league a bad movie"
sentiment_4 = "Star wars was a brilliant film"
sentiment_5 = "Suicide squad has amazing cinematography"

print(Key_categories[np.argmax(model.predict([TensorOut(sentiment_1)]))])
print(Key_categories[np.argmax(model.predict([TensorOut(sentiment_2)]))])
print(Key_categories[np.argmax(model.predict([TensorOut(sentiment_3)]))])
print(Key_categories[np.argmax(model.predict([TensorOut(sentiment_4)]))])
print(Key_categories[np.argmax(model.predict([TensorOut(sentiment_5)]))])
