##################################################################
# Defines two classes, together to make an object detector model #
# MSFE: Multi-Scale Feature Extractor                            #
# SOD: Simple Object Detector (uses MSFE as first layers)        #
##################################################################

import tensorflow as tf
import numpy as np

class MSFE_Model(tf.keras.Model):
    ''' 
    Input/Output: (bs, h, w, c) -> (bs, h, w, f) 
    bs: batch size, c: channels, f: num features
    '''
    def __init__(self, num_features):
        super(Model, self).__init__()

        # Params
        num_filters = 8

        # Small scale feature extractor layers
        self.fe_small = []
        self.fe_small.append(tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3,3), padding='same', activation='relu'))
        self.fe_small.append(tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3,3), padding='same', activation='relu'))
        self.fe_small.append(tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3,3), padding='same', activation='relu'))

        # Medium scale feature extractor layers
        self.fe_medium = []
        self.fe_medium.append(tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3,3), padding='same', activation='relu'))
        self.fe_medium.append(tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3,3), padding='same', activation='relu'))
        self.fe_medium.append(tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3,3), padding='same', activation='relu'))

        # Large scale feature extractor layers
        self.fe_large = []
        self.fe_large.append(tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3,3), padding='same', activation='relu'))
        self.fe_large.append(tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3,3), padding='same', activation='relu'))
        self.fe_large.append(tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3,3), padding='same', activation='relu'))

        # Reconstruction layers (just for pre-training)
        self.recon = []
        self.recon.append(tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3,3), padding='same', activation='relu'))
        self.recon.append(tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3,3), padding='same', activation='relu'))
        self.recon.append(tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3,3), padding='same', activation='relu'))

    def call(x, training=False):
        ''' 
        Trains as an autoencoder but provides features during inference
        '''
        h = x
     
        # Small scale feature extractor layers
        for layer in self.fe_small:
            h = layer(h)

        features_small = h

        # Medium scale feature extractor layers
        for layer in self.fe_medium:
            h = layer(h)

        features_medium = h

        # Large scale feature extractor layers
        for layer in self.fe_large:
            h = layer(h)

        features_large = h

        if(training):
            features_all = tf.concat([features_small, features_medium, features_large])
            # Reconstruct the original image
            for layer in self.recon:
                features_all = layer(features_all)

            return features_all
        else:
            return features_small, features_medium, features_large
        
#class SOD_Model:

