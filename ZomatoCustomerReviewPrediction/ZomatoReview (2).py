# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 23:20:36 2019

@author: LALITHA
"""

import pandas as pd
import numpy as np
import pickle

dataset=pd.read_excel(r"C:\Users\LALITHA\Documents\Vaishu AI\Zomato2k.xlsx",delimiter="\t",quoting=3)
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
data=[]
for i in range(0,2000):
         review=dataset["Review"][i]
         review=re.sub("[^a-zA-Z]"," ",review)
         review=review.lower()
         review=review.split()
         review=[ps.stem(word) for word in review if not word in set (stopwords.words("english"))]
         review= " ".join(review)
         data.append(review)
         
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=2000)
x=cv.fit_transform(data).toarray()
with open('CountVectorizer','wb') as file:
    cv=pickle.dump(cv,file)

y=dataset.iloc[:,1].values     

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
         
from keras.models import Sequential
from keras.layers import Dense
model=Sequential()         
model.add(Dense(input_dim=1336,init="uniform",activation="relu",output_dim=6))
model.add(Dense(output_dim=5,activation="relu",init="uniform"))
model.add(Dense(output_dim=1,activation="sigmoid",init="uniform"))
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
model.fit(x_train,y_train,epochs=150,batch_size=32)
model.save("zomato.h5")
y_pred=model.predict(x_test)

y_pred=(y_pred>0.5)

        