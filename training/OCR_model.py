import sys
import keras
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Add, Activation, Concatenate, Input
from keras.models import Model

def create_model():
	#create network
	input_layer = Input(shape=(None,None,3),name='input')
	
	#convulution the layer
	conv_layer = Conv2D(16, 3, activation='linear', padding='same', strides=(1,1))(input_layer)
	conv_layer = BatchNormalization()(conv_layer)
	conv_layer = Activation('relu')(conv_layer)

	conv_layer = Conv2D(16, 3, activation='linear', padding='same', strides=(1,1))(conv_layer)
	conv_layer = BatchNormalization()(conv_layer)
	conv_layer = Activation('relu')(conv_layer)
	conv_layer = MaxPooling2D(pool_size=(2,2))(conv_layer)
	
	conv_layer = Conv2D(32, 3, activation='linear', padding='same', strides=(1,1))(conv_layer)
	conv_layer = BatchNormalization()(conv_layer)
	conv_layer = Activation('relu')(conv_layer)
	conv_layer = MaxPooling2D(pool_size=(2,2))(conv_layer)

	conv_layer = Conv2D(64, 3, activation='linear', padding='same', strides=(1,1))(conv_layer)
	conv_layer = BatchNormalization()(conv_layer)
	conv_layer = Activation('relu')(conv_layer)
	conv_layer = MaxPooling2D(pool_size=(2,2))(conv_layer)

	conv_layer = Conv2D(64, 3, activation='linear', padding='same', strides=(1,1))(conv_layer)
	conv_layer = BatchNormalization()(conv_layer)
	conv_layer = Activation('relu')(conv_layer)
	conv_layer = MaxPooling2D(pool_size=(2,2))(conv_layer)

	conv_layer = Conv2D(128, 3, activation='linear', padding='same', strides=(1,1))(conv_layer)
	conv_layer = BatchNormalization()(conv_layer)
	conv_layer = Activation('relu')(conv_layer)

	conv_layer_prob = Conv2D(2, 3, activation='softmax', padding='same')(conv_layer)
	conv_layer_box = Conv2D(6, 3, activation='linear' , padding='same')(conv_layer)
	conv_layer = Concatenate(3)([conv_layer_prob,conv_layer_box])

	return Model(inputs=input_layer,outputs=conv_layer)

print 'Creating OCR model'
model = create_model()
#print 'Finished'
model.save_weights('%s.h5' % sys.argv[1])
print'Model has been saved as %s' % sys.argv[1]

