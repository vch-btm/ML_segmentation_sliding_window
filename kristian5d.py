# coding=utf-8

from __future__ import print_function, division
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "0";

import sys
import time

import keras
from keras import backend as K
import tensorflow as tf

config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 4})
sess = tf.Session(config=config)
K.set_session(sess)

from keras.layers import Conv2D, Input, Layer, BatchNormalization
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adam, Optimizer
from keras.callbacks import TensorBoard
import numpy as np
import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt
from shutil import copyfile

from anna_help import *

#########################################
#########################################

# numDataTrain = 8
batch_size = 2 ** 10
# batch_size_res = 512
num2display = 3

numEpochs = 100000
# label_type = "COLL"
label_type = "PG"

#########################################
#########################################

# numDataValid = 4  # numDataTrain // 10
# numDataTest = 4  # numDataTrain // 10
path = ""
shuffleData = not True
randomTxt = ""
netType = "5d_neu"
versionNr = ""

pad_size = -1
winSize = -1

#########################################
#########################################

ID = "{}_{}".format(time.strftime("kb%Y%m%d_%H%M"), label_type)


########################################################################################################################


def build_discriminator(txt_name, set_values=False):
    global predictor, winSize, pad_size

    def d_layer(layer_input, filters, numStrides=1, f_size=3):
        d = Conv2D(filters, kernel_size=f_size, strides=numStrides)(layer_input)
        d = Mish()(d)
        # d = BatchNormalization()(d)
        return d

    d0 = Input(shape=(None, None, 1))
    # d0 = Input(shape=(117, 117, 1))
    # d = BatchNormalization()(d0)
    d = d0

    d = d_layer(d, 16, 2, 3)
    # d = d_layer(d, 16, 1, 5)
    # d = d_layer(d, 16, 1, 5)
    #d = d_layer(d, 16, 2, 3)
        # d = d_layer(d, 16, 2, 3)
    # d = d_layer(d, 32, 2, 3)
    # d = d_layer(d, 16, 1, 5)
    # d = d_layer(d, 16, 1, 3)
    # d = d_layer(d, 16, 2, 3)
    # d = d_layer(d, 16, 1, 3)
    # d = d_layer(d, 16, 1, 3)
    # d = d_layer(d, 16, 2, 3)
    d = d_layer(d, 16, 2, 3)
    d = d_layer(d, 32, 2, 3)
    d = d_layer(d, 64, 1, 3)
    # d = d_layer(d, 256, 2, 3)

    d = Conv2D(1, kernel_size=3, activation="sigmoid")(d)

    dis = Model(d0, d, name=txt_name)

    if set_values:
        predictor, temp = build_predictor(dis)
        winSize = temp + 1
        pad_size = temp // 2

        print("values set: padsize = {}, winSize = {}".format(pad_size, winSize))

        dis._layers[0].batch_input_shape = (None, winSize, winSize, 1)
        with keras.utils.CustomObjectScope({'Mish': Mish}):
            dis2 = keras.models.model_from_json(dis.to_json())
            # dis2.set_weights(dis.get_weights())

        dis = dis2

    return dis


########################################################################################################################


def sample_images(text):
    os.makedirs('{}/{}/imagesAll'.format(label_type, ID), exist_ok=True)
    r, c = len(displayImagesNrs), 5
    factor = 8

    fig, axs = plt.subplots(r, c, figsize=(22.5 * c, 20 * r))
    predictor.set_weights(dis.get_weights())

    for i in range(r):
        actImgNr = displayImagesNrs[i]

        img_x = x_test[actImgNr].reshape((img_rows + 2 * pad_size, img_cols + 2 * pad_size))
        img_y = y_test[actImgNr]

        fake_y = predict_image(img_x, predictor, 2 * pad_size, factor)

        img_x = img_x[pad_size:img_rows + pad_size, pad_size:img_cols + pad_size]

        for j in range(c):
            if r == 1:
                ax = axs[j]
            else:
                ax = axs[i, j]

            ax.axis('off')

            switcher = {
                0: [img_x, {'cmap': plt.get_cmap('gray')}],
                1: [img_y, {}],  # {'vmin': 0, 'vmax': 1}],
                2: [(fake_y[:, :] >= 0.5), {'vmin': 0, 'vmax': 1}],
                3: [1.0 * img_y - (fake_y[:, :] >= 0.5), {'vmin': -1, 'vmax': 1, 'cmap': plt.get_cmap('bwr', 3)}],
                4: [fake_y, {'vmin': 0, 'vmax': 1}],
                # 5: [fake_y, {}]
                # 3: [img_y - fake_y, {'vmin': -1, 'vmax': 1, 'cmap': plt.get_cmap('bwr', 5)}]
            }

            temp = switcher.get(j)
            pcm = ax.imshow(temp[0], **temp[1])

            fig.colorbar(pcm, ax=ax, ticks=np.arange(-1, 1 + 1))

    fig.savefig("{}/{}/imagesAll/{}_{}.png".format(label_type, ID, text, time.strftime("kb%Y%m%d_%H%M")), bbox_inches="tight")
    plt.close()


def train(epochs, sample_interval, batch_size=1):
    num_total = int(numDataTrain * realSize / batch_size) * batch_size

    best_loss = np.infty

    for epoch in range(epochs):

        for batch_i, (data_x, data_y) in enumerate(load_batch(num_total, batch_size, x_train, y_set=y_train, get_y=True, shuffle=True)):
            # sample_images("{}_{}".format(epoch, batch_i))

            loss = dis.train_on_batch(data_x, data_y)  # , class_weight=class_weights)
            loss2 = dis.test_on_batch(data_x, data_y)

            if not (epoch == 0 and batch_i == 0) and batch_i % (10 * sample_interval) == 0:
                print("{} {} {} {} {} {}".format(epoch, batch_i, ID, loss, loss2, time.ctime(time.time())))

                if type(loss) is list:
                    ploss = loss[0]
                    ploss2 = loss2[0]
                else:
                    ploss = loss
                    ploss2 = loss2

                write_log(tensorboard, ['loss'], [ploss], epoch)
                write_log(tensorboard, ['loss2'], [ploss2], epoch)
                write_log(tensorboard, ['zzz_diff'], [ploss - ploss2], epoch)
                dis.save(os.path.join(path, "{}{}/{}/{}_dis_{}_{}.h5".format(path, label_type, ID, ID, netType, versionNr)))
                if ploss2 < best_loss:
                    best_loss = ploss2
                    dis.save(os.path.join(path, "{}{}/{}/{}_dis_{}_{}_best.h5".format(path, label_type, ID, ID, netType, versionNr)))

            if not (epoch == 0 and batch_i == 0) and batch_i % (1000 * sample_interval) == 0:
                print("start generation")
                t0 = time.time()
                sample_images("{}_{}".format(epoch, batch_i))
                t1 = time.time()
                print("image generation: {}".format(t1 - t0))


def load_batch_rot(num_total, bs, x_set, y_set=None, get_y=False, shuffle=False):
    def func0(A):
        return A.T

    def func1(A):
        return np.rot90(A, 1)

    def func2(A):
        return np.rot90(A, 2)

    def func3(A):
        return np.rot90(A, -1)

    def func4(A):
        return np.fliplr(A)

    def func5(A):
        return np.flipud(A)

    def func6(A):
        return np.copy(A)

    n_batches = num_total // bs

    order = np.arange(num_total)

    switcher = {
        0: func0,
        1: func1,
        2: func2,
        3: func3,
        4: func4,
        5: func5,
        6: func6
    }

    if shuffle:
        np.random.shuffle(order)

    for i in range(n_batches):
        virtInd = order[i * bs:(i + 1) * bs]

        cnt = 0

        data_x = np.zeros((bs, winSize, winSize, 1))
        data_y = None

        if get_y:
            data_y = np.zeros((batch_size, 1, 1, 1))

        for actInd in virtInd:
            actImg, coords = divmod(actInd, realSize)
            actRow, actCol = divmod(coords, img_rows)

            rand_int = np.random.randint(7)
            # rand_int = 0

            data_x[cnt] = switcher.get(rand_int, lambda *_: "ERROR: not in list")(x_set[actImg][actRow:actRow + winSize, actCol:actCol + winSize]).reshape(winSize, winSize, 1)
            # data_x[cnt] = x_set[actImg][actRow:actRow + winSize, actCol:actCol + winSize].reshape(winSize, winSize, 1)

            if data_x[cnt].max() == 0.0:
                continue

            if get_y:
                data_y[cnt] = y_set[actImg, actRow, actCol]

            cnt += 1

        if not cnt == n_batches:
            data_x = data_x[:cnt]
            data_y = data_y[:cnt]

        yield data_x, data_y


def load_batch(num_total, bs, x_set, y_set=None, get_y=False, shuffle=False):
    n_batches = num_total // bs

    order = np.arange(num_total)

    if shuffle:
        np.random.shuffle(order)

    for i in range(n_batches):
        virtInd = order[i * bs:(i + 1) * bs]

        cnt = 0

        data_x = np.zeros((bs, winSize, winSize, 1))
        data_y = None

        if get_y:
            data_y = np.zeros((batch_size, 1, 1, 1))

        for actInd in virtInd:
            actImg, coords = divmod(actInd, realSize)
            actRow, actCol = divmod(coords, img_rows)

            rand_int = np.random.randint(7)
            # rand_int = 0

            data_x[cnt] = x_set[actImg][actRow:actRow + winSize, actCol:actCol + winSize].reshape(winSize, winSize, 1)

            if data_x[cnt].max() == 0.0:
                continue

            if get_y:
                data_y[cnt] = y_set[actImg, actRow, actCol]

            cnt += 1

        if not cnt == n_batches:
            data_x = data_x[:cnt]
            data_y = data_y[:cnt]

        yield data_x, data_y


########################################################################################################################


dis = build_discriminator("dis", set_values=True)

if os.path.isdir("/home/vhorak/Desktop"):
    pathData = "/home/vhorak/Desktop/Anna/"
else:
    pathData = "/home/horakv/Schreibtisch/Anna/"

x_train, y_train, x_valid, y_valid, x_test, y_test = load_data_anna(pathData, label_type, shuffleData=False)

# xAllData = np.load("{}images.npy".format(pathData))
# yAllData = np.load("{}labels_{}.npy".format(pathData, label_type)) // 2

numDataTrain, img_rows, img_cols = x_train.shape

realSize = img_rows * img_cols

batch_size_res = 1
temp = realSize

while temp % 2 == 0:
    batch_size_res *= 2
    temp = temp // 2

#########################

x_train = np.pad(x_train, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)), 'symmetric')
x_valid = np.pad(x_valid, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)), 'symmetric')
x_test = np.pad(x_test, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)), 'symmetric')

#########################

os.makedirs('{}/{}'.format(label_type, ID), exist_ok=True)
file_name = os.path.basename(sys.argv[0])
copyfile(file_name, "{}/{}/{}".format(label_type, ID, file_name))
copy_help_file("{}/{}/".format(label_type, ID))

# optimizer = Adam(0.00002, 0.7)
# optimizer = Adam(0.00002)  # , 0.75)
optimizer = RAdam2()
dis.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
# dis.compile(loss='kullback_leibler_divergence', optimizer=optimizer, metrics=['accuracy'])


# dis.compile(loss='mse', optimizer=optimizer)

# comp_weights = [1, 2]  # für PG]
# comp_weights = [35.85635246/32, 0.50707087]  # für PG]
# dis.compile(loss=weighted_categorical_crossentropy(comp_weights), optimizer=optimizer)
# dis.compile(loss='mse', optimizer=optimizer)  # , metrics=['accuracy'])


# dis.compile(optimizer=optimizer, loss=weighted_cross_entropy(2.0))

tensorboard = TensorBoard(log_dir="{}logs/{}/{}{}_{}_{}{}".format(path, label_type, img_rows, randomTxt, ID, netType, versionNr))
tensorboard.set_model(dis)
dis.summary()
plot_model(dis, to_file='{}/{}/{}_dis.png'.format(label_type, ID, ID), show_shapes=True)

########################################################################################################################
########################################################################################################################

displayImagesNrs = [0, 1, 2, 3]

train(epochs=numEpochs, sample_interval=10, batch_size=batch_size)
