from tensorflow import keras
import numpy as np

(kunren_gazou, kunren_label), (kensho_gazou,
                               kensho_label) = keras.datasets.fashion_mnist.load_data()
kunren_label = keras.utils.to_categorical(
    kunren_label, num_classes=None)
kensho_label = keras.utils.to_categorical(
    kensho_label, num_classes=None)

kunren_gazou = kunren_gazou/255.0
kensho_gazou = kensho_gazou/255.0
model = keras.models.load_model('Jijyou_heikinn_ep20.h5')

kensho_loss, kensho_acc = model.evaluate(kensho_gazou, kensho_label, verbose=2)

print('\n')
print('kensho loss:', kensho_loss)
print('kensho_accuracy:', kensho_acc)

kenshosample = np.expand_dims(kensho_gazou[300], 0)
yosoku_kekka = model.predict(kenshosample)
print(yosoku_kekka[0])
print(np.argmax(yosoku_kekka[0]))
