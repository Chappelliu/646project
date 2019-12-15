import sys
import cv2
import numpy
import traceback
import darknet.python.darknet as dn
from os.path 				import splitext, basename
from glob				import glob
from darknet.python.darknet 		import detect

class Label:

	def __init__(self,cl=-1,tl=numpy.array([0.,0.]),br=numpy.array([0.,0.]),prob=None):
		self.__tl = tl
		self.__br = br
		self.__cl = cl
		self.__prob = prob

	def prob(self): return self.__prob
	def tl(self): return self.__tl
	def br(self): return self.__br
	def cl(self): return self.__cl

def IOU_labels(l1,l2):
	return IOU(l1.tl(),l1.br(),l2.tl(),l2.br())

def IOU(tl1,br1,tl2,br2):
	wh1,wh2 = br1-tl1,br2-tl2
	assert((wh1>=.0).all() and (wh2>=.0).all())
	
	intersection_wh = numpy.maximum(numpy.minimum(br1,br2) - numpy.maximum(tl1,tl2),0.)
	intersection_area = numpy.prod(intersection_wh)
	area1,area2 = (numpy.prod(wh1),numpy.prod(wh2))
	union_area = area1 + area2 - intersection_area;
	return intersection_area/union_area


output_dir = 'test/output'

ocr_weights = 'traineddata/646-ocr.weights'
ocr_netcfg  = 'traineddata/646-ocr.cfg'
ocr_dataset = 'traineddata/646-ocr.data'
#using dn to load the trained data and make the ocr
ocr_net  = dn.load_net(ocr_netcfg, ocr_weights, 0)
ocr_meta = dn.load_meta(ocr_dataset)
#get all the test.png file under the output directory
imgs_paths = glob('test/output/*test.png')

print 'Start OCR operation...'
for i,img_path in enumerate(imgs_paths):
	print '\tScanning %s' % img_path
	bname = basename(splitext(img_path)[0])
	R,(width,height) = detect(ocr_net, ocr_meta, img_path ,thresh=0.4, nms=None)
	if len(R):
		x = numpy.array([width,height],dtype=float)
		L  = []
		for r in R:
			center = numpy.array(r[2][:2])/x
			x2 = (numpy.array(r[2][2:])/x)*.5
			L.append(Label(ord(r[0]),tl=center-x2,br=center+x2,prob=r[1]))

		SelectedLabels = []
		L.sort(key=lambda l: l.prob(),reverse=True)
		for label in L:
			non_overlap = True
			for sel_label in SelectedLabels:
				if IOU_labels(label,sel_label) > 0.45:
					non_overlap = False
					break
			if non_overlap:
				SelectedLabels.append(label)
		L = SelectedLabels
		L.sort(key=lambda x: x.tl()[0])
		pl_num = ''.join([chr(l.cl()) for l in L])
		print '\t\tLP: %s' % pl_num
	else:

		print 'No characters found'

