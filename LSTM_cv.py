import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from nltk.corpus import stopwords
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv("reviews.csv", encoding = "utf-8")
reviews = df['Görüş'].values.astype('U') # Görüş: Reviews Column

df["Durum"].replace({"Olumlu": 1, "Olumsuz": 0}, inplace=True) # Olumlu: Positive || Olumsuz: Negative
sentiments = df['Durum'] # Durum: Sentiments Column

_stopwords = list(stopwords.words('Turkish'))
reviews_cleaned = []
for text in reviews:
    j = str(text)
    j = j.split()
    j = [word for word in j if word not in _stopwords]
    j = ' '.join(j)
    reviews_cleaned.append(j)
            
X = np.array(reviews_cleaned)
y = np.array(sentiments).reshape(-1,1)

num_words = 10000
tok = Tokenizer(num_words=num_words)
tok.fit_on_texts(X)
sequences = tok.texts_to_sequences(X)
X_new = sequence.pad_sequences(sequences, padding = "pre", truncating ="pre")

k=5
folds = []
kfold=KFold(k,True,42)
for train, test in kfold.split(X_new, y):
    folds.append([train, test])
    
accuracies = []
losses = []
conf_matrices = []

i = 0
for j in range(len(folds)):
    X_train = X_new[folds[i][0]]
    X_test = X_new[folds[i][1]]

    y_train = y[folds[i][0]]
    y_test = y[folds[i][1]]

    model = Sequential()
    model.add(Embedding(num_words, 128, input_length=X_new.shape[1]))
    model.add(LSTM(196, recurrent_dropout = 0.3, dropout = 0.3))
    model.add(Dense(1, activation = "sigmoid"))
        
    model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=4, batch_size=32, validation_split = 0.10)

    loss,accuracy = model.evaluate(X_test, y_test)
    print("loss: ", loss)
    print("accuracy: ", accuracy)
    accuracies.append(accuracy)
    losses.append(loss)
    
    predy = model.predict(X_test)
    predy_list = [1 if i > 0.5 else 0 for i in predy]
    conf_matrices.append(confusion_matrix(y_test, predy_list))
    i += 1
    
print(sum(accuracies)/k)
print(sum(losses)/k)
print(sum(conf_matrices))
