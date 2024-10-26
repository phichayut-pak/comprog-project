# -*- coding: utf-8 -*-
"""Insurance.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1oVvU6d6qfxh1WVkw5OHZd5aOOyUeGtXJ
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# from google.colab import drive
# drive.mount('/content/drive',force_remount=True)

raw_data = pd.read_csv('insurance.csv')
print(raw_data)

raw_data.columns

raw_data.loc[raw_data['bmi'].isnull()]

raw_data.replace(to_replace=dict(female=0, male=1), inplace=True)
raw_data.replace(to_replace=dict(no=0, yes=1), inplace=True)
raw_data.replace(to_replace=dict(northwest=1, northeast=2,southeast=3,southwest=4), inplace=True)
raw_data = raw_data.dropna()
raw_data

plt.plot(raw_data['region'],raw_data['charges'])

plt.plot(raw_data['age'],raw_data['charges'])

"""# ใช้ 5 ตัวแปรในการ predict Charges


"""

input = raw_data[['age','sex','bmi','children','smoker']]
input

input = input.to_numpy()
input

output = raw_data['charges'].to_numpy()
output

model1 = Sequential()
model1.add(Dense(1, input_dim=5))
model1.add(Dense(3))
model1.add(Dense(3))
model1.add(Dense(3))
model1.add(Dense(1))

model1.compile(optimizer=tf.optimizers.Adam(learning_rate=0.05),loss='mean_absolute_error')
history = model1.fit(input, output, epochs=100)
plt.plot(history.history['loss'])

predictions = model1.predict(input)
plt.plot(output,label = 'Real',color='purple')
plt.plot(predictions,label='Predictions',color = 'g')
plt.legend()
plt.title("y from data and predictions")
fig = plt.gcf()
fig.set_size_inches(30, 10.5)

plt.xlabel("input")
plt.ylabel("charges")
plt.show()

predictions = model1.predict(input)
plt.plot(output[:100],label = 'Real',color='purple')
plt.plot(predictions[:100],label='Predictions',color = 'g')
plt.legend()
plt.title("y from data and predictions")
fig = plt.gcf()
fig.set_size_inches(30, 10.5)

plt.xlabel("input")
plt.ylabel("charges")
plt.show()

"""# ใช้ค่า BMI ในการ predict Charges



"""

input = raw_data['bmi'].to_numpy()
input

output = raw_data['charges'].to_numpy()
output

model2 = Sequential()
model2.add(Dense(1, input_dim=1))
model2.add(Dense(1))
model2.compile(optimizer=tf.optimizers.Adam(learning_rate=0.05),loss='mean_absolute_error')
history = model2.fit(input, output, epochs=100)
plt.plot(history.history['loss'])

predictions = model2.predict(input)
plt.plot(output,label = 'Real',color='purple')
plt.plot(predictions,label='Predictions',color = 'g')
plt.legend()
plt.title("charges from data and predictions")
fig = plt.gcf()
fig.set_size_inches(30, 10.5)

plt.xlabel("person")
plt.ylabel("charges")
plt.show()

predictions = model2.predict(input)
plt.plot(output[:100],label = 'Real',color='purple')
plt.plot(predictions[:100],label='Predictions',color = 'g')
plt.legend()
plt.title("first 100 data of charges from data and predictions")
fig = plt.gcf()
fig.set_size_inches(30, 10.5)

plt.xlabel("input")
plt.ylabel("charges")
plt.show()

"""จะเห็นว่าความแม่นยำไม่แม่นเท่าใช้ตัวแปร 5 ตัวแปร

# ใช้ 6 ตัวแปรในการ predict Charges (เพิ่ม region)
"""

input = raw_data[['age','sex','bmi','children','smoker','region']].to_numpy()
input

output = raw_data['charges'].to_numpy()
output

model3 = Sequential()
model3.add(Dense(1, input_dim=6))
model3.add(Dense(4))
model3.add(Dense(4))
model3.add(Dense(1))

model3.compile(optimizer=tf.optimizers.Adam(learning_rate=0.05),loss='mean_absolute_error')
history = model3.fit(input, output, epochs=100)
plt.plot(history.history['loss'])

predictions = model3.predict(input)
plt.plot(output,label = 'Real',color='purple')
plt.plot(predictions,label='Predictions',color = 'g')
plt.legend()
plt.title("y from data and predictions")
fig = plt.gcf()
fig.set_size_inches(30, 10.5)

plt.xlabel("input")
plt.ylabel("charges")
plt.show()

predictions = model3.predict(input)
plt.plot(output[:100],label = 'Real',color='purple')
plt.plot(predictions[:100],label='Predictions',color = 'g')
plt.legend()
plt.title("y from data and predictions")
fig = plt.gcf()
fig.set_size_inches(30, 10.5)

plt.xlabel("input")
plt.ylabel("charges")
plt.show()

"""จะเห็นว่า Graph ของ 5 ตัวแปร กับ 6 ตัวแปร เหมือนกันเลย ซึ่ง
อาจจะมาจากการที่ region ไม่ได้ส่งผลต่อ charges
"""