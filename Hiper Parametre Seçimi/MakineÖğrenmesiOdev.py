#!/usr/bin/env python
# coding: utf-8

# In[97]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# İris veri setini yüklüyoruz

# In[100]:


iris=pd.read_csv("./IRIS.csv")


# Veriyi özellikler ve hedef olarak ayırın.

# In[102]:


X = iris.drop('species', axis=1)
y = iris['species'] 


# Şimdi iki özelliğe göre veri setimizi görselleştiriyoruz.

# In[99]:


plt.scatter(X[y == 'setosa']['sepal_length'], X[y == 'setosa']['sepal_width'], label='Iris setosa', color='r', marker='o')
plt.scatter(X[y == 'versicolor']['sepal_length'], X[y == 'versicolor']['sepal_width'], label='Iris versicolor', color='g', marker='x')
plt.scatter(X[y == 'virginica']['sepal_length'], X[y == 'virginica']['sepal_width'], label='Iris virginica', color='b', marker='s')

plt.xlabel('Çiçek Yaprak Uzunluğu (cm)')
plt.ylabel('Çiçek Yaprak Genişliği (cm)')
plt.legend(loc='upper right')
plt.title('Çiçek Yapraklarının Uzunluğu ve Genişliği')
plt.show()


# Giriş ve çıkış değerlerinin boyutlarına bakalım.

# In[77]:


print(X.shape)
print(y.shape)


# Veriyi eğitim ve test setlerine yazdıralım.

# In[80]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Şimdi veriyi ölçeklendirelim

# In[81]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Lojistik regresyonun modelini oluşturuyoruz ve eğitiyoruz.

# In[104]:


model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train) #modeli eğittik
lr = LogisticRegression(max_iter=1000)
lr.fit(X, y)


# Parametre ve bias değerlerimizi görelim.

# In[107]:


print("Parametre: ", lr.coef_)
print("Bias: ", lr.intercept_)


# Test verileri üzerinde tahminler yapıyoruz.

# In[83]:


y_pred = model.predict(X_test)


# Modeli diske kaydediyoruz.

# In[84]:


import joblib

# Eğitilmiş modeli kaydetme
joblib.dump(model, 'egitilmis_model.pkl')


# Modeli yüklüyoruz.

# In[85]:


loaded_model = joblib.load('egitilmis_model.pkl')
y_pred = loaded_model.predict(X_test)


# Model performansını değerlendiriyoruz.

# In[86]:


accuracy=accuracy_score(y_test, y_pred)
confusion=confusion_matrix(y_test, y_pred)
report=classification_report(y_test, y_pred)


# In[87]:


print(f"Model Doğruluk: {accuracy:.2f}")
print("Confusion Matrix:\n",confusion)
print("Sınıflandırma Raporu:\n",report)


# In[ ]:





# In[ ]:




