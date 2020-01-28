'''
Created on October, 2018
by: Meshal
'''

import numpy as np
np.random.seed(1337)

from keras.models import Model
from keras.layers import multiply, Input, Dense, Activation
import keras

    
# Attentive Autoencoder
def AttAE_module(input_size, dimension):
    print ("Attentive Autoencoder is used")
    print ("Arch:", dimension*8, dimension*4, dimension*2, dimension)
    max_features = input_size

    model_input=Input(shape=(max_features,),dtype='float32',name='model_input')

    nn = Dense(dimension*8, use_bias=False)(model_input) #1st hidden layer
    nn = keras.layers.normalization.BatchNormalization()(nn)
    nn = Activation("relu")(nn)
    nn = Dense(dimension*4, use_bias=False)(nn) #2nd hidden layer
    nn = keras.layers.normalization.BatchNormalization()(nn)
    nn = Activation("relu")(nn)
    nn = Dense(dimension*2, use_bias=False)(nn) #3rd hidden layer
    nn = keras.layers.normalization.BatchNormalization()(nn)
    nn = Activation("relu")(nn)

    attention_probs = Dense(dimension*2, activation='softmax', name='attention_probs')(nn)
    attention = multiply([nn, attention_probs], name='attention')

    z = Dense(dimension, use_bias=False, name='z')(attention) #z
    z = keras.layers.normalization.BatchNormalization()(z)
    z = Activation("relu")(z)

    nn = Dense(dimension*2, use_bias=False)(z)
    nn = keras.layers.normalization.BatchNormalization()(nn)
    nn = Activation("relu")(nn)
    nn = Dense(dimension*4, use_bias=False)(nn)
    nn = keras.layers.normalization.BatchNormalization()(nn)
    nn = Activation("relu")(nn)
    nn = Dense(dimension*8, use_bias=False)(nn)
    nn = keras.layers.normalization.BatchNormalization()(nn)
    nn = Activation("relu")(nn)
    model_output = Dense(max_features, activation='sigmoid', name='output')(nn)

    train_model= Model(inputs=model_input, outputs=model_output)
    eval_model = Model(inputs=model_input, outputs=z)
    train_model.compile(optimizer='rmsprop', loss=keras.losses.binary_crossentropy)

    return train_model, eval_model


##################################################################
def load_model(model, model_path):
    model.load_weights(model_path)

def save_model(model, model_path, isoverwrite=True):
    model.save_weights(model_path, isoverwrite)

def train(train_model, X_train, X_train_hat, nb_epoch):
    history = train_model.fit(X_train,X_train_hat,verbose=2, epochs=nb_epoch) 
    return history
    
def get_z_layer(model, X_train):
    Z = model.predict(X_train, verbose=0)
    return Z
    

