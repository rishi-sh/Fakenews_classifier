# -*- coding: utf-8 -*-

import pandas as pd

df =pd.read_csv('train.csv') #r Read the file
df = df.dropna() # dropping NaN values
x=df.drop('label',axis=1) #dropping label from x
y=df['label'] # adding label column to y

import tensorflow as tf #importing tensorflow library


#lib require to implement LSTM
from tensorflow.keras.layers import Embedding

from tensorflow.keras.preprocessing.sequence import pad_sequences #helps to make the input lenght fixed 
#we can add zero padded in thr start of the NN or at the end

from tensorflow.keras.models import Sequential 

from tensorflow.keras.preprocessing.text import one_hot# helps to convert sentences in one hot representation
#by giving a vocab size

from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense #helps to give the prob 

#define your vocab size
voc_size =5000

#one hot representation
message=x.copy()
message.reset_index(inplace=True)

#importing nlp lib
import nltk
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
#data cleaning
ps=PorterStemmer()

corpus=[]
for i in range(len(message)):
    review = re.sub('[^a-zA-Z]', ' ', message['title'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
#converting corpus to onr hot rep
onehot_repr=[one_hot(words,voc_size)for words in corpus] 
onehot_repr

#embedding rep
#sentence lenght is selected based on the  max lenght of the sentence in you data
sent_length=20
embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
#'pre' says that if the sen_length are not equal to 20 then they will add zeros in the start side to make it to 20
#and for 'post' zeros will be added at the end  
print(embedded_docs)

embedded_docs[0]

## Creating model
# in embedding layer we take in input and convert into specific num of feature vectors
#bsed on the num of feature it will give you the output,here we are taking it as 40
embedding_vector_features=40
model=Sequential()#creating a seq layer
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))#adding a seq layer
model.add(LSTM(3))#passing a 1 lstm layer which has 100 neurons to seq layer
model.add(Dense(1,activation='sigmoid'))#adding a dence layer   
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())

len(embedded_docs),y.shape

import numpy as np
X_final=np.array(embedded_docs) #converting embedded do into an array   
y_final=np.array(y)
X_final.shape,y_final.shape

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.20, random_state=42)

### Finally Training
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=20,batch_size=64)

#addding a hypeparameter tuining tech
#here we are using a droupout layer
#droupout layer is added in between
from tensorflow.keras.layers import Dropout
embedding_vector_features=40
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(Dropout(0.3))
model.add(LSTM(3))
model.add(Dropout(0.4))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

y_pred=model.predict_classes(X_test)
from sklearn.metrics import confusion_matrix
con=confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
acc =accuracy_score(y_test,y_pred)

