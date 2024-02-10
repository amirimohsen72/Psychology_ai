import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense


df = pd.read_csv('dep.csv')
df.head()

plt.subplots(figsize=(9,9))
sns.heatmap(df.corr(),annot=True)

x=df.drop('dep',axis=1)
y=df.dep

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)


model=Sequential()
model.add(Dense(64,input_dim=11,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(3,activation='sigmoid'))

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),optimizer='adam',metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train,epochs=500,batch_size=10 )


# man=np.array([[1,3,2,33,0,0,1,2,2,1,2]])
safa=np.array([[1,4,3,44,1,2,1,0,2,0,1]])
out = model.predict(safa)

print(out)

h = np.array(out[0])
bar = ('low','meduim', 'high')
plt.bar(bar,h)

plt.show()
