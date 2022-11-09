import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

(kunren_gazou, kunren_label), (kensho_gazou,
                               kensho_label) = keras.datasets.fashion_mnist.load_data()
kunren_label = keras.utils.to_categorical(
    kunren_label, num_classes=None)
kensho_label = keras.utils.to_categorical(
    kensho_label, num_classes=None)

kunren_gazou = kunren_gazou/255.0
kensho_gazou = kensho_gazou/255.0

model = keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)), keras.layers.Dense(
    100, activation='sigmoid'), keras.layers.Dense(
    130, activation='sigmoid'), keras.layers.Dense(10, activation='softmax')])

model.summary()

model.compile(optimizer='SGD', loss='mean_squared_error', metrics=['accuracy'])

model.fit(kunren_gazou, kunren_label, epochs=10)

model.save('Jijyou_heikinn_ep10.h5')

kensho_loss, kensho_acc = model.evaluate(kensho_gazou, kensho_label, verbose=2)

print('\n')
print('Kensho loss (sonshitsu:', kensho_loss)
print('kensho accuracy Seikai_ritsu:', kensho_acc)
