from tkinter import *
from tkinter import filedialog
import os
import tkinter as tk
from PIL import Image, ImageTk
from os.path import join, exists
from matplotlib import image
from matplotlib import pyplot
from numpy import asarray
import numpy as np

# ==============================================
# MODEL APP
# ==============================================

import tensorflow as tf

from tensorflow.keras.applications.vgg16 import VGG16 #load transfer learning vgg16 model
from tensorflow.keras.applications.resnet import ResNet50 #load transfer learning resnet model
from tensorflow.keras.applications.inception_v3 import InceptionV3 #load transfer learning inceptionv3 model
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2 #load transfer learning inception reset v2 model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2 #load transfer learning mobilenet v2 model

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda, Reshape
from tensorflow.keras.layers import Conv2D, SeparableConv2D, Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import Add, Concatenate
from tensorflow.keras.optimizers import Adam

from enum import Enum #impor library enum berfungsi sebagai alias pengganti nilai variabel

class_name = ['Granit','Limestone','Marmer','Motif','Mozaik','Teraso']
K_class = len(class_name)

#mendeklarasikan class Model dengan input enum
class MODEL(Enum):
	TF_VGG16 = 1
	TF_ResNet50 = 2
	TF_InceptionV3 = 3
	TF_InceptionResNetV2 = 4
	TF_MobileNetV2 = 5

# select model to use
opt_model = MODEL.TF_VGG16
# opt_model = MODEL.TF_ResNet50
# opt_model = MODEL.TF_InceptionV3 #menggunakan transfer learning model inception v3
# opt_model = MODEL.TF_InceptionResNetV2
# opt_model = MODEL.TF_MobileNetV2

# define rescale normalization
norm_ratio = 1./255 #mendefinisikan rasio normalisasi
norm_offset = 0  #mendefinisikan offset normalisasi
 
# number of epoch
epochs = 100
batch_size = 32

# define the input size
input_size = (225, 225, 3) #definisikan input size sesuai dengan resolusi gambar (225 x 225)
if opt_model == MODEL.TF_InceptionV3 or opt_model == MODEL.TF_InceptionResNetV2: #jika opsi model inception v3 atau inception resnet v2
	input_size = (225, 225, 3) #mendefinisikan input size (225,225,3) seperti deklarasi awal

# build conv -> batch normalization -> activation function
def build_conv_bnorm_fun(input_layer, nfilters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu'): #mendeklarasikan layer konvolusi
	add_layer = Conv2D(filters=nfilters, kernel_size=kernel_size, strides=strides, padding='same')(input_layer) #mengatur layer konvolusi
	add_layer = BatchNormalization()(add_layer) #mengatur layer batch normalisasi
	output_layer = Activation(activation)(add_layer) #menambahkan fungsi aktivasi
        
	return output_layer #mengembalikan nilai dari setting layer konvolusi

def build_dense_bnorm_fun(input_layer, nfilters=512, activation='relu'): #mendeklarasikan layer neural network
	add_layer = Dense(nfilters)(input_layer) #menambahkan layer dense (neural network)
	add_layer = BatchNormalization()(add_layer) #menambahkan batch normalisasi
	output_layer = Activation(activation)(add_layer) #menambahkan fungsi aktivasi

	return output_layer #mengembalikan nilai dari setting layer dense (neural network)

# build pre-processing layer
def build_preprocessing_model(input_layer, input_size, norm_ratio=1./255.0, norm_offset=0): #mendeklarasikan preprocessing model
	output_layer = layers.experimental.preprocessing.Rescaling(norm_ratio, norm_offset)(input_layer) #menambahkan layer untuk preprocessing (pada kasus ini berupa rescaling / normalisasi)

	return output_layer #mengembalikan nilai dari setting layer preprocessing

# tensorflow VGG16
def build_vgg16_tf(input_layer, input_size, K): #setup untuk model vgg16
	# load base model
	base_model = VGG16(include_top=False, pooling='avg', weights=None, input_shape=input_size)

	add_layer = base_model(input_layer, training=True) #menambahkan layer dari base model

	# FC layer 1: dense -> bnorm -> relu -> dropout 
	add_layer = build_dense_bnorm_fun(add_layer, nfilters=4096, activation='relu') #menambahkan layer (dense) neural network yang sudah kita deklarasikan diatas

	# FC layer 2: dense -> bnorm -> relu -> dropout 
	add_layer = build_dense_bnorm_fun(add_layer, nfilters=4096, activation='relu') #menambahkan layer (dense) neural network yang sudah kita deklarasikan diatas

	# output layer: dense -> softmax 
	output_layer = Dense(K, activation='softmax', name='main_model')(add_layer) #menambahkan layer dense sebagai output layer dengan fungsi aktivasi softmax

	return output_layer #mengembalikan semua setup dari model vgg16

# tensorflow ResNet50
def build_resnet50_tf(input_layer, input_size, K): #setup untuk model resnet50
	# load base model
	base_model = ResNet50(include_top=False, pooling='avg', weights=None, input_shape=input_size)

	add_layer = base_model(input_layer, training=True) #menambahkan layer dari base model

	# output layer: dense -> softmax 
	output_layer = Dense(K, activation='softmax', name='main_model')(add_layer) #menambahkan layer dense sebagai output layer dengan fungsi aktivasi softmax

	return output_layer #mengembalikan semua setup dari model resnet50

# tensorflow InceptionV3
def build_inceptionV3_tf(input_layer, input_size, K): #setup untuk model inception v3
	# load base model
	base_model = InceptionV3(include_top=False, pooling='avg', weights=None, input_shape=input_size)

	add_layer = base_model(input_layer, training=True) #menambahkan layer dari base model

	# output layer: dense -> softmax 
	output_layer = Dense(K, activation='softmax', name='main_model')(add_layer) #menambahkan layer dense sebagai output layer dengan fungsi aktivasi softmax

	return output_layer #mengembalikan semua setup dari model inception v3

# tensorflow InceptionResNetV2
def build_inceptionresnetV2_tf(input_layer, input_size, K): #setup untuk model inception resnet v2
	# load base model
	base_model = InceptionResNetV2(include_top=False, pooling='avg', weights=None, input_shape=input_size)

	add_layer = base_model(input_layer, training=True) #menambahkan layer dari base model

	# output layer: dense -> softmax 
	output_layer = Dense(K, activation='softmax', name='main_model')(add_layer) #menambahkan layer dense sebagai output layer dengan fungsi aktivasi softmax

	return output_layer #mengembalikan semua setup dari model inception resnet v2

# tensorflow MobileNetV2
def build_mobilenetV2_tf(input_layer, input_size, K): #setup untuk model mobilenet v2
	# load base model
	base_model = MobileNetV2(include_top=False, pooling='avg', weights=None, input_shape=input_size)

	add_layer = base_model(input_layer, training=True) #menambahkan layer dari base model

	# output layer: dense -> softmax 
	output_layer = Dense(K, activation='softmax', name='main_model')(add_layer) #menambahkan layer dense sebagai output layer dengan fungsi aktivasi softmax

	return output_layer #mengembalikan semua setup dari model mobilenet v2

# input layer
inputs = Input(shape=input_size)

global main_model

# select model from the list of pre-trained models
if opt_model == MODEL.TF_VGG16: #jika opsi yang dipilih model dari vgg16
	input_model = tf.keras.applications.vgg16.preprocess_input(inputs) #load pre-trained model dari vgg16
	main_model = build_vgg16_tf(input_model, input_size, K_class) #load model yang telah dideklarasikan di atas
elif opt_model == MODEL.TF_ResNet50: #jika opsi yang dipilih model dari resnet 50
	input_model = tf.keras.applications.resnet50.preprocess_input(inputs) #load pre-trained model dari resnet50
	main_model = build_resnet50_tf(input_model, input_size, K_class) #load model yang telah dideklarasikan di atas
elif opt_model == MODEL.TF_InceptionV3: #jika opsi yang dipilih model dari inception v3
	input_model = tf.keras.applications.inception_v3.preprocess_input(inputs) #load pre-trained model dari inception v3
	main_model = build_inceptionV3_tf(input_model, input_size, K_class) #load model yang telah dideklarasikan di atas
elif opt_model == MODEL.TF_InceptionResNetV2: #jika opsi yang dipilih model dari resnet v2
	input_model = tf.keras.applications.inception_resnet_v2.preprocess_input(inputs) #load pre-trained model dari resnet v2
	main_model = build_inceptionresnetV2_tf(input_model, input_size, K_class) #load model yang telah dideklarasikan di atas
elif opt_model == MODEL.TF_MobileNetV2: #jika opsi yang dipilih model dari mobilenet v2
	input_model = tf.keras.applications.mobilenet_v2.preprocess_input(inputs) #load pre-trained model dari mobilenet v2
	main_model = build_mobilenetV2_tf(input_model, input_size, K_class) #load model yang telah dideklarasikan di atas

def build_model():
	global main_model
	# build final model
	main_model = Model(inputs=inputs, outputs=main_model, name='Ceramic_model') #membuat model yang akan digunakan, dan menamakan dengan ceramic_model

	# initialize the optimizer and compile the model
	init_learning_rate = 1e-4 #membuat learning rate 0.0001
	opt_optimizer = Adam(learning_rate=init_learning_rate) #membuat optimizer menggunakan adam
	main_model.compile(optimizer=opt_optimizer, loss='sparse_categorical_crossentropy', loss_weights=1.0, metrics=['accuracy']) #kompilasi learning rate dan optimizer adam

# ==============================================
# TKINTER APP
# ==============================================

root_path = "model/"

data = list()

def classify():
	global data, main_model, class_name
	try:
		result_text.set('Please wait...	')

		X = np.expand_dims(data, axis=0)
		# print(X.shape)
		try:
			build_model()
		except:
			print('model has been loaded')

		checkpoint_filepath = join(root_path, 'keramik_model.h5')
		main_model.load_weights(checkpoint_filepath)

		prediction_scores = main_model.predict(X)
		score = np.argmax(prediction_scores)

		print(prediction_scores)
		print(np.max(prediction_scores) * 100)

		# if score == 0:
		# 	result_text.set('Granit')
		# elif score == 1:
		# 	result_text.set('Limestone')
		# elif score == 2:
		# 	result_text.set('Marmer')
		# elif score == 3:
		# 	result_text.set('Motif')
		# elif score == 4:
		# 	result_text.set('Mozaik')
		# elif score == 5:
		# 	result_text.set('Teraso')

		result_text.set(class_name[score])

	except Exception as e:
		result_text.set("Error to classification")
		print("Can't classify this image")
		print(e)


def load_image():
	global data
	fln = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select Image File", filetypes=(("All Files", "*.*"), ("JPG File", "*.jpg"), ("PNG File", "*.png")))
	try:
		result_text.set('')
		img = Image.open(fln)
		img.thumbnail((225, 225))
		img = ImageTk.PhotoImage(img)
		lbl.configure(image=img)
		lbl.image = img

		classify_img = Image.open(fln)
		data = asarray(classify_img)
	except:
		result_text.set("Can't Load Image")
		print("Can't Load Image File")


root = Tk()

#create icon in window
icon = PhotoImage(file='asset/icon.png')
root.iconphoto(False, icon)

frm = Frame(root)
frm.pack(side=BOTTOM, padx=15, pady=15)

title = tk.Label(root, text='KLASIFIKASI KERAMIK', font="Raleway")
title.pack(side=tk.TOP, pady=15)

lbl = Label(root)
lbl.pack()

result_text = tk.StringVar()
result = tk.Label(root, textvariable=result_text, font="Raleway")
result.pack(side=tk.BOTTOM)

browseBtn = Button(frm, text="Upload Gambar", command=load_image)
browseBtn.pack(side=tk.LEFT)

exitBtn = Button(frm, text="Cek Keramik", command=classify)
exitBtn.pack(side=tk.LEFT, padx=10)

root.title("Keramik Classification")
root.geometry("300x350")
root.mainloop()