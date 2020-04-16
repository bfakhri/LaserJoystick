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
    def __init__(self):
        super(MSFE_Model, self).__init__()

        # Params
        num_filters = 8
        self.stops  = np.array([1, 3, 5])
        num_layers = np.max(self.stops)+1

        # Multi-scale feature extractor layers
        self.fe_lyrs = []
        for i in range(num_layers):
            self.fe_lyrs.append(tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(5,5), strides=(2,2), padding='same', activation='relu'))

        # Reconstruction layers
        self.recon_lyr = tf.keras.layers.Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')

    def call(self, x, reconstruct=True):
        ''' 
        Trains as an autoencoder but provides features during inference
        '''
        h = x
        saved_h = []
        print('Orig: ', h.shape)
        # Small scale feature extractor layers
        for idx,layer in enumerate(self.fe_lyrs):
            h = layer(h)
            if(np.any(idx == self.stops)):
                saved_h.append(h)
                print(idx, h.shape, '\tSaved!')
            else:
                print(idx, h.shape)

        if(reconstruct):
            # Concat all the saved features for reconstruction
            saved_h_rs = []
            for feats in saved_h:
                print('Resizing: ', feats.shape[1:3], ' to: ', x.shape[1:3])
                saved_h_rs.append(tf.image.resize(feats, x.shape[1:3]))
                print('new size: ', saved_h_rs[-1].shape)
            h = tf.concat(saved_h_rs, axis=-1)
            print('Concated: ', h.shape)
            # Perform reconstruction
            h = self.recon_lyr(h)
            print('Reconstructed: ', h.shape)
            return h 
        else:
            return saved_h


        
class SOD_Model:
    ''' 
    Object detector
    '''
    def __init__(self):
        super(SOD_Model, self).__init__()

        # Model Params
        num_filters = 8
        output_size = 10

        # Load the Feature Extractor Model
        self.fe_model = MSFE_Model()
        MSFE_Model.load('msfe_model.extension')

        # Simple object detection layers
        self.od_lyrs = []
        self.od_lyrs.append(tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3,3), padding='same', activation='relu'))
        self.od_lyrs.append(tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3,3), padding='same', activation='relu'))
        self.od_lyrs.append(tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3,3), padding='same', activation='relu'))


