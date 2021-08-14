## for data
import pandas as pd
import numpy as np
import string
## for processing
import re
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from nltk import word_tokenize
from nltk.corpus import stopwords
## for deep learning
from sklearn import model_selection
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras import Sequential
from keras.initializers.initializers_v2 import Constant



#load dataset
dtf = pd.read_csv("spam_or_not_spam.csv") #passing the dataset into a pandas dataframe



## rename columns
dtf = dtf.rename(columns={"label":"y", "email":"text"})


dtf["text"] = dtf["text"].apply(str)


#functions for data preprocessing
def remove_URL(text):
    url = re.compile(r"https?://\S+|www\.\S+")
    return url.sub(r"", text)


def remove_html(text):
    html = re.compile(r"<.*?>")
    return html.sub(r"", text)

def remove_emoji(string):
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", string)


def remove_punct(text):
    table = str.maketrans("", "", string.punctuation)
    return text.translate(table)

dtf["text"] = dtf.text.map(lambda x: remove_URL(x))
dtf["text"] = dtf.text.map(lambda x: remove_html(x))
dtf["text"] = dtf.text.map(lambda x: remove_emoji(x))
dtf["text"] = dtf.text.map(lambda x: remove_punct(x))

stop = set(stopwords.words("english"))


def remove_stopwords(text):
    text = [word.lower() for word in text.split() if word.lower() not in stop]

    return " ".join(text)

dtf["text"] = dtf["text"].map(remove_stopwords)

#function for corpus creation
def create_corpus_tk(df):
    corpus = []
    for text in dtf["text"]:
        words = [word.lower() for word in word_tokenize(text)]
        corpus.append(words)
    return corpus

corpus = create_corpus_tk(dtf)
num_words = len(corpus)

#dataset split
train, test = model_selection.train_test_split(dtf, test_size=0.25)

max_len = 50

#create the vocabulary of indices
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(train["text"])

#text to indices
train_sequences = tokenizer.texts_to_sequences(train["text"])

#padding sequences
train_padded = pad_sequences(
    train_sequences, maxlen=max_len, truncating="post", padding="post"
)
#text to indices
test_sequences = tokenizer.texts_to_sequences(test["text"])

#padding sequences
test_padded = pad_sequences(
    test_sequences, maxlen=max_len, padding="post", truncating="post"
)
word_index = tokenizer.word_index

#creating embedding dictionary
embedding_dict = {}
with open('glove.6B.100d.txt', encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vectors = np.asarray(values[1:], "float32")
        embedding_dict[word] = vectors
f.close()

#initilize embedding matrix
num_words = len(word_index) + 1
embedding_matrix = np.zeros((num_words, 100))

#values in embedding matrix
for word, i in word_index.items():
    if i < num_words:
        emb_vec = embedding_dict.get(word)
        if emb_vec is not None:
            embedding_matrix[i] = emb_vec


#building LSTM RNN
model = Sequential()

model.add(
    layers.Embedding(
        num_words,
        100,
        embeddings_initializer=Constant(embedding_matrix),
        input_length=max_len,
        trainable=False,
    )
)
model.add(layers.LSTM(100, dropout=0.1))
model.add(layers.Dense(1, activation="sigmoid"))

#metrics functions
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


#compile model
model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['acc',f1_m,precision_m, recall_m])

#train model
history = model.fit(
    train_padded,
    train["y"],
    epochs=20,
    validation_data=(test_padded, test["y"]),
    verbose=1,
)

# evaluate model
loss, accuracy, f1_score, precision, recall = model.evaluate(test_padded, test["y"], verbose=1)
print('Accuracy: %f' % accuracy)
print('Precision: %f' % precision)
print('Recall: %f' % recall)
print('F1 score: %f' % f1_score)


