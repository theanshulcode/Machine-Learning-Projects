#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


dataset=pd.read_csv(r"C:\Users\hp\Desktop\Stock Predictor\trainset.xls",index_col="Date",parse_dates=True)


# In[3]:


dataset.head()


# In[4]:


dataset['Open'].plot(figsize=(20,8))


# In[5]:


dataset.info()


# In[6]:


#converting the datatype from int64 to float64
dataset["Volume"] = dataset["Volume"].astype(float)


# In[7]:


dataset.info()


# In[8]:


#6 days mean
dataset.rolling(6).mean().head(20)


# In[9]:


dataset['Open'].plot(figsize=(20,8))
dataset.rolling(window=30).mean()['Close'].plot()


# In[10]:


training_set=dataset['Open']
training_set=pd.DataFrame(training_set)


# In[11]:


# scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scale = sc.fit_transform(training_set)


# In[12]:


# Created Data Sets
xtrain = []
ytrain = []
for i in range(60,1258):
    xtrain.append(training_set_scale[i-60:i,0])
    ytrain.append(training_set_scale[i,0])
xtrain = np.array(xtrain)
ytrain = np.array(ytrain)

xtrain = np.reshape(xtrain, (xtrain.shape[0],xtrain.shape[1],1))


# In[13]:


from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout


# In[14]:


#RNN
regressor = Sequential()


# In[15]:


# First LSDM layer
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (xtrain.shape[1],1)))
regressor.add(Dropout(0.2))
#sec LSDM layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
#third LSDM layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
#forth LSDM layer
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
#outputs
regressor.add(Dense(units=1))


# In[16]:


#compiler
regressor.compile(optimizer='adam', loss='mean_squared_error')

#fillting
regressor.fit(xtrain,ytrain, epochs=100, batch_size=32)


# In[17]:


dataset_test = pd.read_csv(r"C:\Users\hp\Desktop\Stock Predictor\testset.xls",index_col="Date",parse_dates=True)


# In[18]:


real_price = dataset_test.iloc[:,1:2].values


# In[19]:


dataset_test.head(10)


# In[20]:


dataset_test.info()


# In[21]:


dataset_test["Volume"] = dataset_test["Volume"].astype(float)


# In[22]:


dataset_test.info()


# In[23]:


test_set=dataset_test['Open']
test_set=pd.DataFrame(test_set)


# In[24]:


dataset_total = pd.concat((dataset['Open'],dataset_test['Open']),axis=0)
inputs = dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
xtest=[]
for i in range(60,185):
    xtest.append(inputs[i-60:i,0])
xtest = np.array(xtest)
xtest = np.reshape(xtest, (xtest.shape[0],xtest.shape[1],1))
predicted_price = regressor.predict(xtest)
predicted_price = sc.inverse_transform(predicted_price)


# In[25]:


predicted_price = pd.DataFrame(predicted_price)
predicted_price.info()


# In[26]:


plt.plot(real_price, color = 'black',label = 'Real Stock Price')
plt.plot(predicted_price, color = 'red', label = 'Predicted Stock Price' )
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


# In[ ]:




