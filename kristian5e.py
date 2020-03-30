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
int_parts = 2 ** 6

numEpochs = 100000
#label_type = "COLL"
label_type = "PG"

#########################################
#########################################

# numDataValid = 4  # numDataTrain // 10
# numDataTest = 4  # numDataTrain // 10
path = ""
shuffleData = not True
randomTxt = ""
netType = "5e_Unet"
versionNr = ""

pad_size = int_parts // 2
winSize = int_parts

#########################################
#########################################

ID = "{}_{}".format(time.strftime("kb%Y%m%d_%H%M"), label_type)


########################################################################################################################


def build_unet(txt_name):
    if not True:
        downConv = convComb
    else:
        downConv = conv

    if not True:
        upConv = convUp
    else:
        upConv = convT

    d0 = Input(shape=img_shape)

    num_rep = int(np.log2(img_rows))
    num_parts1 = 0
    num_parts2 = 0
    num_dep_rep = 3
    factor = 1

    max_dil = 1

    # (Crop(3, 0, mult_h // 2 + 1)

    d_list = [d0]

    for _ in range(num_rep):
        for _ in range(num_dep_rep):
            # d_list.append(downConv(d_list[-1], 4 * factor, 3, 1, parts=num_parts1, dil_max=max_dil))
            d_list.append(conv(d_list[-1], 4 * factor, 3, 1, parts=num_parts1, dil_max=1))

        d_list.append(downConv(d_list[-1], 2 * factor, 3, 2, parts=num_parts1, dil_max=max_dil))

    d_last = d_list.pop()
    n_list = [d_last]
    n = d_last

    for _ in range(1):
        n_list.append(conv(n, 8 * factor, 1, 1, parts=0))
        n = Concatenate()(n_list)

    d_list.append(n)
    d_list.append(d_last)

    for _ in range(num_rep):
        d = Concatenate()([d_list.pop(), d_list.pop()])
        d_list.append(upConv(d, 4 * factor, 3, 2, parts=num_parts2))
        for _ in range(num_dep_rep):
            d = Concatenate()([d_list.pop(), d_list.pop()])
            # d_list.append(upConv(d, 4 * factor2, 3, 1, parts=num_parts2))
            d_list.append(conv(d, 4 * factor, 3, 1, parts=num_parts2))

    for _ in range(1):
        d1 = d_list.pop()
        d2 = conv(d1, 1 * factor, 3, 1, parts=num_parts2)
        d_list.append(Concatenate()([d1, d2]))

    fin = conv(d_list.pop(), 1, 1, 1, parts=num_parts2)

    return Model(d0, fin, name=txt_name)


########################################################################################################################


def sample_images(text):
    os.makedirs('{}/{}/imagesAll'.format(label_type, ID), exist_ok=True)
    r, c = len(displayImagesNrs), 5
    factor = 8

    fig, axs = plt.subplots(r, c, figsize=(22.5 * c, 20 * r))
    # predictor.set_weights(dis.get_weights())

    for i in range(r):
        actImgNr = displayImagesNrs[i]

        img_x = x_test[actImgNr].reshape((img_rows + 2 * pad_size, img_cols + 2 * pad_size))
        img_y = y_test[actImgNr]





        # batch-generator hier auf einzelnes bild anwenden


        fake_y = predict_image(img_x, unet, 2 * pad_size, factor)





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

            loss = unet.train_on_batch(data_x, data_y)  # , class_weight=class_weights)
            loss2 = unet.test_on_batch(data_x, data_y)

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
                unet.save(os.path.join(path, "{}{}/{}/{}_unet_{}_{}.h5".format(path, label_type, ID, ID, netType, versionNr)))
                if ploss2 < best_loss:
                    best_loss = ploss2
                    unet.save(os.path.join(path, "{}{}/{}/{}_unet_{}_{}_best.h5".format(path, label_type, ID, ID, netType, versionNr)))

            if not (epoch == 0 and batch_i == 0) and batch_i % (100 * sample_interval) == 0:
                print("start generation")
                t0 = time.time()
                sample_images("{}_{}".format(epoch, batch_i))
                t1 = time.time()
                print("image generation: {}".format(t1 - t0))


def load_batch(num_total, bs, x_set, y_set=None, get_y=False, shuffle=False):
    n_batches = num_total // bs

    int_parts_half = int_parts // 2

    order = np.arange(num_total)

    if shuffle:
        np.random.shuffle(order)

    for i in range(n_batches):
        virtInd = order[i * bs:(i + 1) * bs]

        cnt = 0

        data_x = np.zeros((bs, winSize, winSize, 1))
        data_y = None

        if get_y:
            data_y = np.zeros((bs, winSize, winSize, 1))

        for actInd in virtInd:
            actImg, pos = divmod(actInd, num_img_parts * num_img_parts)
            pos_x, pos_y = divmod(pos, num_img_parts)

            data_x[cnt] = x_set[actImg][pos_x * int_parts_half:(pos_x + 1) * int_parts_half,
                          pos_y * int_parts_half:(pos_y + 1) * int_parts_half].reshape(winSize, winSize, 1)

            if data_x[cnt].max() == 0.0:
                continue

            if get_y:
                data_y[cnt] = y_set[actImg][pos_x * int_parts_half:(pos_x + 1) * int_parts_half,
                              pos_y * int_parts_half:(pos_y + 1) * int_parts_half].reshape(winSize, winSize, 1)

            cnt += 1

        if not cnt == n_batches:
            data_x = data_x[:cnt]
            data_y = data_y[:cnt]

        yield data_x, data_y


########################################################################################################################


unet = build_unet("unet")

if os.path.isdir("/home/vhorak/Desktop"):
    pathData = "/home/vhorak/Desktop/Anna/"
else:
    pathData = "/home/horakv/Schreibtisch/Anna/"

x_train, y_train, x_valid, y_valid, x_test, y_test = load_data_anna(pathData, label_type, shuffleData=False)

# xAllData = np.load("{}images.npy".format(pathData))
# yAllData = np.load("{}labels_{}.npy".format(pathData, label_type)) // 2

numDataTrain, img_rows, img_cols = x_train.shape

realSize = img_rows * img_cols
num_img_parts = 2 * numDataTrain / int_parts - 1

batch_size_res = 1
temp = realSize

while temp % 2 == 0:
    batch_size_res *= 2
    temp = temp // 2

#########################

x_train = np.pad(x_train, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)), 'symmetric')
x_valid = np.pad(x_valid, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)), 'symmetric')
x_test = np.pad(x_test, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)), 'symmetric')

y_train = np.pad(y_train, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)), 'symmetric')
y_valid = np.pad(y_valid, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)), 'symmetric')
y_test = np.pad(y_test, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)), 'symmetric')

#########################

os.makedirs('{}/{}'.format(label_type, ID), exist_ok=True)
file_name = os.path.basename(sys.argv[0])
copyfile(file_name, "{}/{}/{}".format(label_type, ID, file_name))
copy_help_file("{}/{}/".format(label_type, ID))

# optimizer = Adam(0.00002, 0.7)
# optimizer = Adam(0.00002)  # , 0.75)
optimizer = RAdam2()
unet.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
# dis.compile(loss='kullback_leibler_divergence', optimizer=optimizer, metrics=['accuracy'])


# dis.compile(loss='mse', optimizer=optimizer)

# comp_weights = [1, 2]  # für PG]
# comp_weights = [35.85635246/32, 0.50707087]  # für PG]
# dis.compile(loss=weighted_categorical_crossentropy(comp_weights), optimizer=optimizer)
# dis.compile(loss='mse', optimizer=optimizer)  # , metrics=['accuracy'])


# dis.compile(optimizer=optimizer, loss=weighted_cross_entropy(2.0))

tensorboard = TensorBoard(log_dir="{}logs/{}/{}{}_{}_{}{}".format(path, label_type, img_rows, randomTxt, ID, netType, versionNr))
tensorboard.set_model(unet)
unet.summary()
plot_model(unet, to_file='{}/{}/{}_dis.png'.format(label_type, ID, ID), show_shapes=True)

########################################################################################################################
########################################################################################################################

displayImagesNrs = [0, 1, 2, 3]

train(epochs=numEpochs, sample_interval=10, batch_size=batch_size)
