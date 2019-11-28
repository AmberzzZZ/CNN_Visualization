from keras.applications.vgg16 import VGG16
from keras.layers import Input, GlobalAveragePooling2D, Dense
from keras.models import Model


def flatten_model(input_shape):
    model = VGG16(include_top=True, weights='imagenet', input_shape=input_shape)
    return model


def gap_model(input_shape):
    input = Input(input_shape)
    print(input_shape)
    basemodel = VGG16(include_top=False, weights='imagenet')
    x = basemodel(input)
    x = GlobalAveragePooling2D()(x)
    x = Dense(10, 'softmax')(x)

    model = Model(input, x)

    return model





