# coding: utf-8
# visualize the feature maps for each layer

import cv2
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
import numpy as np
import keras.backend as K


def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col


def visualize_feature_map(img_batch, layer_name=None):
    feature_map = np.squeeze(img_batch, axis=0)
    print(feature_map.shape)         # (h,w,c)

    feature_map_combination = []
    plt.figure()

    num_pic = feature_map.shape[2]
    row, col = get_row_col(num_pic)

    for i in range(0, num_pic):
        feature_map_split = feature_map[:, :, i]
        feature_map_combination.append(feature_map_split)
        plt.subplot(row, col, i + 1)
        plt.imshow(feature_map_split)
        plt.axis('off')
        # plt.title('feature_map_{}'.format(i))

    plt.savefig('feature_map_%s.png' %(layer_name))
    plt.show()

    # 各个特征图按1：1 叠加
    feature_map_sum = sum(ele for ele in feature_map_combination)
    plt.imshow(feature_map_sum)
    plt.savefig("feature_map_sum_%s.png" %(layer_name))


if __name__ == "__main__":
    img = cv2.imread('tux_hacking.jpg')
    img_batch = np.expand_dims(img, axis=0)

    model = VGG16(include_top=False, weights='imagenet', input_shape=img.shape)
    layer_name = 'block5_conv3'
    func = K.function(inputs=model.inputs, outputs=[model.get_layer(layer_name).output])
    conv_img = func([img_batch])[0]
    print(conv_img.shape)

    visualize_feature_map(conv_img[...,:], layer_name)



