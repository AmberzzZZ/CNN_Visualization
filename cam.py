# import cv2


# superimposed_img = cv2.addWeighted(img,0.6,heatmap,0.4,0)

# -*- coding=utf-8 -*-
import cv2
import numpy as np
import keras.backend as K
from models import *
K.set_learning_phase(1) #set learning phase


img = cv2.imread('tux_hacking.jpg')
img = np.expand_dims(img, axis=0)

model = gap_model(img.shape)
model.summary()
# model.load_weights("tobemodified", by_name=True)

pred = model.predict(img)
# print(pred.shape)

# class_idx = np.argmax(pred[0])
# class_output = model.output[:,class_idx]
# last_conv_layer = model.get_layer("block5_conv3")
# gap_weights = model.get_layer("global_average_pooling2d_1")

# grads = K.gradients(class_output,gap_weights.output)[0]
# iterate = K.function([model.input],[grads,last_conv_layer.output[0]])
# pooled_grads_value, conv_layer_output_value = iterate([img])
# print("pooled_grads_value.shape:  ", pooled_grads_value.shape)
# pooled_grads_value = np.squeeze(pooled_grads_value,axis=0)
# for i in range(512):
#     conv_layer_output_value[:,:,i] *= pooled_grads_value[i]

# heatmap = np.mean(conv_layer_output_value, axis=-1)
# heatmap = np.maximum(heatmap,0)#relu激活。
# heatmap /= np.max(heatmap)
# #
# img = cv2.imread(img_path)
# img = cv2.resize(img,dsize=(224,224),interpolation=cv2.INTER_NEAREST)
# # img = img_to_array(image)
# heatmap = cv2.resize(heatmap,(img.shape[1],img.shape[0]))
# heatmap = np.uint8(255 * heatmap)
# heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)
# superimposed_img = cv2.addWeighted(img,0.6,heatmap,0.4,0)
# cv2.imshow('Grad-cam',superimposed_img)
# cv2.waitKey(0)




