import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import numpy as np

class Autoencoder:
    def __init__(self, image_dim):
        self.encoding_dim = 4
        self.image_dim = image_dim

        """input_img = keras.Input(shape=(self.image_dim,)) # , kernel_regularizer='l1', bias_regularizer='l2'
        encoded = layers.Dense(self.encoding_dim*20, activation='linear', kernel_regularizer='l1', bias_regularizer='l1')(input_img)
        #encoded = layers.Dense(self.encoding_dim*18, activation='linear', kernel_regularizer='l1', bias_regularizer='l1')(encoded)
        encoded = layers.Dense(self.encoding_dim*16, activation='linear', kernel_regularizer='l1', bias_regularizer='l1')(encoded)
        #encoded = layers.Dense(self.encoding_dim*14, activation='linear', kernel_regularizer='l1', bias_regularizer='l1')(encoded)
        encoded = layers.Dense(self.encoding_dim*12, activation='linear', kernel_regularizer='l1', bias_regularizer='l1')(encoded)
        #encoded = layers.Dense(self.encoding_dim*10, activation='linear', kernel_regularizer='l1', bias_regularizer='l1')(encoded)
        encoded = layers.Dense(self.encoding_dim*8, activation='linear', kernel_regularizer='l1', bias_regularizer='l1')(encoded)
        #encoded = layers.Dense(self.encoding_dim*6, activation='linear', kernel_regularizer='l1', bias_regularizer='l1')(encoded)
        encoded = layers.Dense(self.encoding_dim*4, activation='linear', kernel_regularizer='l1', bias_regularizer='l1')(encoded)
        #encoded = layers.Dense(self.encoding_dim*2, activation='relu')(input_img)
        encoded = layers.Dense(self.encoding_dim, activation='sigmoid', kernel_regularizer='l1', bias_regularizer='l1')(encoded)
        decoded = layers.Dense(self.encoding_dim*4, activation='linear', kernel_regularizer='l1', bias_regularizer='l1')(encoded)
        #decoded = layers.Dense(self.encoding_dim*6, activation='linear', kernel_regularizer='l1', bias_regularizer='l1')(decoded)
        decoded = layers.Dense(self.encoding_dim*8, activation='linear', kernel_regularizer='l1', bias_regularizer='l1')(decoded)
        #decoded = layers.Dense(self.encoding_dim*10, activation='linear', kernel_regularizer='l1', bias_regularizer='l1')(decoded)
        decoded = layers.Dense(self.encoding_dim*12, activation='linear', kernel_regularizer='l1', bias_regularizer='l1')(decoded)
        #decoded = layers.Dense(self.encoding_dim*14, activation='linear', kernel_regularizer='l1', bias_regularizer='l1')(decoded)
        decoded = layers.Dense(self.encoding_dim*16, activation='linear', kernel_regularizer='l1', bias_regularizer='l1')(decoded)
        #decoded = layers.Dense(self.encoding_dim*18, activation='linear', kernel_regularizer='l1', bias_regularizer='l1')(decoded)
        decoded = layers.Dense(self.encoding_dim*20, activation='linear', kernel_regularizer='l1', bias_regularizer='l1')(decoded)
        decoded = layers.Dense(self.image_dim, activation='linear', kernel_regularizer='l1', bias_regularizer='l1')(decoded)"""

        input_img = keras.Input(shape=(self.image_dim,)) # , kernel_regularizer='l1', bias_regularizer='l2'
        encoded = layers.Dense(self.encoding_dim*20, activation='linear')(input_img)
        #encoded = layers.Dense(self.encoding_dim*18, activation='linear')(encoded)
        encoded = layers.Dense(self.encoding_dim*16, activation='linear')(encoded)
        #encoded = layers.Dense(self.encoding_dim*14, activation='linear')(encoded)
        encoded = layers.Dense(self.encoding_dim*12, activation='linear')(encoded)
        #encoded = layers.Dense(self.encoding_dim*10, activation='linear')(encoded)
        encoded = layers.Dense(self.encoding_dim*8, activation='linear')(encoded)
        #encoded = layers.Dense(self.encoding_dim*6, activation='linear')(encoded)
        encoded = layers.Dense(self.encoding_dim*4, activation='linear')(encoded)
        #encoded = layers.Dense(self.encoding_dim*2, activation='relu')(input_img)
        encoded = layers.Dense(self.encoding_dim, activation='sigmoid')(encoded)
        decoded = layers.Dense(self.encoding_dim*4, activation='linear')(encoded)
        #decoded = layers.Dense(self.encoding_dim*6, activation='linear')(decoded)
        decoded = layers.Dense(self.encoding_dim*8, activation='linear')(decoded)
        #decoded = layers.Dense(self.encoding_dim*10, activation='linear')(decoded)
        decoded = layers.Dense(self.encoding_dim*12, activation='linear')(decoded)
        #decoded = layers.Dense(self.encoding_dim*14, activation='linear')(decoded)
        decoded = layers.Dense(self.encoding_dim*16, activation='linear')(decoded)
        #decoded = layers.Dense(self.encoding_dim*18, activation='linear')(decoded)
        decoded = layers.Dense(self.encoding_dim*20, activation='linear')(decoded)
        decoded = layers.Dense(self.image_dim, activation='linear')(decoded)

        self.autoencoder = keras.Model(input_img, decoded)
        
        self.encoder = keras.Model(input_img, encoded)
        encoded_input = keras.Input(shape=(self.encoding_dim,))
        decoder_layer6 = self.autoencoder.layers[-6]
        decoder_layer5 = self.autoencoder.layers[-5]
        decoder_layer4 = self.autoencoder.layers[-4]
        decoder_layer3 = self.autoencoder.layers[-3]
        decoder_layer2 = self.autoencoder.layers[-2]
        decoder_layer1 = self.autoencoder.layers[-1]
        self.decoder = keras.Model(encoded_input, decoder_layer1(decoder_layer2(decoder_layer3(decoder_layer4(decoder_layer5(decoder_layer6(encoded_input)))))))
        #self.decoder = keras.Model(encoded_input, decoder_layer1(decoder_layer2(decoder_layer3(decoder_layer4(decoder_layer5(decoder_layer6(decoder_layer7(decoder_layer8(decoder_layer9(decoder_layer10(encoded_input)))))))))))

        self.autoencoder.compile(optimizer='Adagrad', loss=keras.losses.MeanAbsoluteError())


    def train(self, data, epochs):
        self.autoencoder.fit(data, data, epochs=epochs, shuffle=True)

    def encode(self, data):
        encoded_data = self.encoder.predict(data)
        return encoded_data

    def decode(self, data):
        decoded_data = self.decoder.predict(data)
        return decoded_data

    def process(self, data):
        encoded_data = self.encoder.predict(data)
        decoded_data = self.decoder.predict(encoded_data)
        return decoded_data

    def display(self, original_data, data):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2)
        ax[0].plot(original_data)
        ax[1].plot(data)
        plt.show()

    def save_self(self, path):
        path_1 = path + "_ae"
        self.autoencoder.save_weights(path_1)

    def load_self(self, path):
        path_1 = path + "_ae"
        self.autoencoder.load_weights(path_1)
        

if __name__ == "__main__":
    imgs = []
    for y in range(100):
        imgs.append([x/100 for x in range(100)])
        imgs.append([np.sin(x/4) for x in range(100)])
        imgs.append([np.cos(x/4) for x in range(100)])
        imgs.append([x/2 for x in range(100)])
    imgs = np.array(imgs)
    ae = Autoencoder(100)
    ae.train(imgs, 10000)
    decoded_data = ae.process(imgs)
    ae.display(imgs[0], decoded_data[0])
    ae.display(imgs[1], decoded_data[1])
    ae.display(imgs[2], decoded_data[2])
    ae.display(imgs[3], decoded_data[3])

