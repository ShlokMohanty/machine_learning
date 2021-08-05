
import keras
# from keras.models import Sequential, Graph
from keras.layers.core import Layer, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, merge
from keras.models import Model
import pdb
# class Identity(Layer):
#     def call(self, x, mask=None):
#         return x

def building_residual_block(input_shape, n_feature_maps, kernel_sizes=None, n_skip=2, is_subsample=False, subsample=None):
    # ***** VERBOSE_PART ***** 
    print ('   - New residual block with')
    print ('      input shape:', input_shape)
    print ('      kernel size:', kernel_sizes)
    # is_expand_channels == True when num_channels increases.
    #    E.g. the very first residual block (e.g. 1->64, 3->128, 128->256, ...)
    is_expand_channels = not (input_shape[0] == n_feature_maps) 
    if is_expand_channels:
        print ('      - Input channels: %d ---> num feature maps on out: %d' % (input_shape[0], n_feature_maps)  )
    if is_subsample:
        print ('      - with subsample:', subsample)
    kernel_row, kernel_col = kernel_sizes
    # set input
    x = Input(shape=(input_shape))
    # ***** SHORTCUT PATH *****
    if is_subsample: # subsample (+ channel expansion if needed)
        shortcut_y = Convolution2D(n_feature_maps, kernel_sizes[0], kernel_sizes[1], 
                                    subsample=subsample,
                                    border_mode='valid')(x)
    else: # channel expansion only (e.g. the very first layer of the whole networks)
        if is_expand_channels:
            shortcut_y = Convolution2D(n_feature_maps, 1, 1, border_mode='same')(x)
        else:
            # if no subsample and no channel expension, there's nothing to add on the shortcut.
            shortcut_y = x
    # ***** CONVOLUTION_PATH ***** 
    conv_y = x
    for i in range(n_skip):
        conv_y = BatchNormalization(axis=1, mode=2)(conv_y)        
        conv_y = Activation('relu')(conv_y)
        if i==0 and is_subsample: # [Subsample at layer 0 if needed]
            conv_y = Convolution2D(n_feature_maps, kernel_row, kernel_col,
                                    subsample=subsample,
                                    border_mode='valid')(conv_y)  
        else:        
            conv_y = Convolution2D(n_feature_maps, kernel_row, kernel_col, border_mode='same')(conv_y)
    # output
    y = merge([shortcut_y, conv_y], mode='sum')
    block = Model(input=x, output=y)
    print ('        -- model was built.')
    return block
    

