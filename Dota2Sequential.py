#import library
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense

#Membaca dataset train
df_train = pd.read_csv('dota2Train copy.csv', delimiter = ';')

#Membaca dataset test
df_test = pd.read_csv('dota2Test copy.csv', delimiter = ';')

#pisahkan X dan Y (input dan output)
X_test = df_test.drop('label', axis=1)
y_test = df_test['label']

X_train = df_test.drop('label', axis=1)
y_train = df_test['label']

#scale input menggunakan standardscaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled

#Model Sequential MLP 2 hidden layers
model = Sequential()
model.add(Dense(units = 116,
                activation = 'sigmoid',
                input_shape = (116,)))

model.add(Dense(units = 68,
                activation = 'sigmoid'))

model.add(Dense(units = 68,
                activation = 'sigmoid'))

model.add(Dense(units=1, 
                activation = 'relu'))

model.compile(loss='binary_crossentropy', 
              optimizer='sgd', 
              metrics=['categorical_accuracy'])

history = model.fit(X_train, y_train, epochs = 100, batch_size = 64)

# evaluasi model test
score = model.evaluate(X_test, y_test)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

# evaluasi model train
scores = model.evaluate(X_train, y_train)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
