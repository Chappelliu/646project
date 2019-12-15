#training alogrithm is modified from ALPR in Unscontrained Scenarios  https://github.com/sergiomsilva/alpr-unconstrained/blob/master/license-plate-ocr.py
import sys
import numpy as np
import cv2
import keras

from random import choice
from os.path import isfile, isdir, basename, splitext
from os import makedirs

from src.keras_utils import save_model, load_model
from src.label import readShapes
from src.loss import loss
from src.utils import image_files_from_folder, show
from src.sampler import augment_sample, labels2output_map
from src.data_generator import DataGenerator, process_data_item

from pdb import set_trace as pause

def process_data_item(data_item,dim,model_stride):
	XX,llp,pts = augment_sample(data_item[0],data_item[1].pts,dim)
	YY = labels2output_map(llp,pts,dim,model_stride)
	return XX,YY

if __name__ == '__main__':

	netname = basename('ocr-trained')
	outdir = 'trained_data'
	iterations = 30000
	batch_size = 32

	#load model from the pre-created model file
	model = load_model('646-ocr')
	xshape = (dim,dim,3)
	inputs  = keras.layers.Input(shape=(dim,dim,3))
	outputs = model(inputs)
	yshape = tuple([s.value for s in outputs.shape[1:]])
	output_dim   = yshape[1]
	model_stride = dim / output_dim
	opt = getattr(keras.optimizers,'Adam')(lr=0.01)
	model.compile(loss=loss, optimizer=opt)

	#read the database from the input folder
	print 'Scanning the data from the input file...'
	Files = image_files_from_folder('input')
	Data = []
	for file in Files:
		labfile = splitext(file)[0] + '.txt'
		if isfile(labfile):
			L = readShapes(labfile)
			I = cv2.imread(file)
			Data.append([I,L[0]])

	dg = DataGenerator(	data=Data, \
				process_data_item_func=lambda x: process_data_item(x,dim,model_stride),\
				xshape=xshape, \
				yshape=(yshape[0],yshape[1],yshape[2]+1), \
				nthreads=2, \
				pool_size=1000, \
				min_nsamples=100 )
	dg.start()

	Xtrain = np.empty((batch_size,dim,dim,3),dtype='single')
	Ytrain = np.empty((batch_size,dim/model_stride,dim/model_stride,2*4+1))

	model_path_final  = '%s/%s_final'  % (outdir,netname)

	for it in range(iterations):

		print 'Iter. %d (of %d)' % (it+1,iterations)

		Xtrain,Ytrain = dg.get_batch(batch_size)
		train_loss = model.train_on_batch(Xtrain,Ytrain)

		print '\tLoss: %f' % train_loss

	print 'Saving model (%s)' % model_path_final
	save_model(model,model_path_final)
