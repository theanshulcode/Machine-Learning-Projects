#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPool1D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, plot_roc_curve


# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[3]:


data = pd.read_csv('creditcard.csv')
data.head()


# In[4]:


data.shape


# In[5]:


data.isnull().sum()


# In[6]:


data.info()


# In[7]:


data['Class'].value_counts()


# ### Balance Dataset

# In[8]:


non_fraud = data[data['Class']==0]
fraud = data[data['Class']==1]


# In[9]:


non_fraud.shape , fraud.shape


# In[10]:


non_fraud = non_fraud.sample(fraud.shape[0])
non_fraud.shape #Making random data and equal data of fraud and non_fraud


# In[11]:


data = fraud.append(non_fraud, ignore_index=True)
data


# In[12]:


data['Class'].value_counts()


# In[13]:


x = data.drop("Class", axis = 1)
y=data['Class']


# In[14]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0, stratify =y)


# In[15]:


x_train.shape, x_test.shape


# In[16]:


scaler =StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[17]:


y_train = y_train.to_numpy() #coverting data set to numpy array
y_test = y_test.to_numpy()


# In[18]:


x_train.shape


# In[19]:


#neural network takes 3D data so we have to convert is to 3D
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)


# In[20]:


x_train.shape , x_test.shape


# ### Build CNN

# In[21]:


epochs=20
model = Sequential()
model.add(Conv1D(32,3, activation ='relu', input_shape = x_train[0].shape))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv1D(64,2, activation ='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
          
model.add(Flatten())
model.add(Dense(64,activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(1,activation = 'sigmoid'))


# In[22]:


model.summary()


# In[23]:


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit(x_train,y_train,batch_size=25,epochs=20, validation_data=(x_test,y_test), verbose = 1)


# In[24]:


def plot_Curve(hisory,epoch):
    epoch_range = range(1,epoch+1)
    plt.plot(epoch_range , hisory.history['accuracy'])
    plt.plot(epoch_range , hisory.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train','test'], loc = 'upper left')
    plt.show()
    
    #Lose valuse
    plt.plot(epoch_range , hisory.history['loss'])
    plt.plot(epoch_range , hisory.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train','test'], loc = 'upper left')
    plt.show()


# In[25]:


plot_Curve(history, epochs)


# ### Adding MaxPool

# In[26]:


epochs=50
model = Sequential()
model.add(Conv1D(32,3, activation ='relu', input_shape = x_train[0].shape))
model.add(BatchNormalization())
model.add(MaxPool1D(2))
model.add(Dropout(0.2))

model.add(Conv1D(64,2, activation ='relu'))
model.add(BatchNormalization())
model.add(MaxPool1D(2))
model.add(Dropout(0.5))
          
model.add(Flatten())
model.add(Dense(64,activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(1,activation = 'sigmoid'))


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit(x_train,y_train,batch_size=25,epochs=epochs, validation_data=(x_test,y_test), verbose = 1)


# In[27]:


plot_Curve(history, epochs)


# In[ ]:




