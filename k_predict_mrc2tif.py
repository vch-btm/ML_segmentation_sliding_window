from __future__ import print_function, division
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import keras
from keras.models import load_model, Input, Model
from keras.layers import Conv2D
import tensorflow as tf
import numpy as np
import time
from tifffile import imsave
import mrcfile
import argparse

from keras.layers import Layer
from keras import backend as K

#########################################################



#########################################################

ID = "{}".format(time.strftime("kb%Y%m%d_%H%M"))

parser = argparse.ArgumentParser(description='predict images')
parser.add_argument('file_in', metavar='IN', type=str, default="labeled_data/380-1_reconstructed.mrc", help="name of input file")
parser.add_argument('-file_out', metavar='OUT', type=str, default="{}.tif".format(ID), help="name of output file")
parser.add_argument("-dis", metavar="DIS", type=str, default='kb20190502_1047_COLL_dis_kb5_1_best.h5', help="name of classifier")
parser.add_argument("-nit", metavar="NIT", type=int, default=50, help="number of iterations (regularization)")
parser.add_argument("-b", metavar="bias", type=float, default=0.5, help="bias (regularization)")
parser.add_argument("-l", metavar="lambda", type=float, default=0.85, help="lambda (regularization)")
parser.add_argument("-debug", metavar="debug", type=bool, default=False, help="activate debug mode")
parser.add_argument('-images', metavar='N', type=int, nargs='+', help='images to process (starting with 0)')

args = parser.parse_args()

#########################################################

file_in = args.file_in
file_out = args.file_out
with keras.utils.CustomObjectScope({'Swish': Swish}):
    dis = load_model(args.dis)
bias = args.b
lamb = args.l
num_tv_it = args.nit
debug_mode = args.debug

mrc = mrcfile.open(file_in)
images = mrc.data.transpose((1, 0, 2))

#########################################

device_name = ""

# device_name = tf.test.gpu_device_name()
# if device_name != '/device:GPU:0':
#     raise SystemError('GPU device not found')

print('Found GPU at: {}'.format(device_name))
print("keras version: {}".format(keras.__version__))
print("tensorflow version: {}".format(tf.__version__))

########################################################################################################################

def predict_image(image, predictor, pad_size, factor=1):
    (num_rows, num_cols) = image.shape

    fake_y = np.zeros(image.shape)

    image = np.pad(image, pad_size//2, 'symmetric')

    ############################################################################

    n_batches = factor * factor
    rows_part = num_rows // factor
    cols_part = num_cols // factor

    w_size = rows_part + pad_size

    x_data = np.zeros((n_batches, w_size, w_size, 1))

    for j in range(n_batches):
        coord_x = j // factor
        coord_y = j % factor

        x_data[j] = image[coord_x * rows_part:(coord_x + 1) * rows_part + pad_size,
               coord_y * cols_part:(coord_y + 1) * cols_part + pad_size].reshape(w_size, w_size, 1)

    ############################################################################

    pred = predictor.predict(x_data)

    for i in range(factor):
        for j in range(factor):
            fake_y[i * rows_part:(i + 1) * rows_part, j * cols_part:(j + 1) * cols_part] = pred[i*factor + j].reshape((rows_part, rows_part))

    return fake_y


########################################################################################################################


def build_predictor(orig_dis):
    num_layers = len(orig_dis.layers)

    dil_rate = 1
    cnt = 0

    d0 = Input(shape=(None, None, 1))

    for i in range(1, num_layers):
        layer = orig_dis.layers[i]
        config = layer.get_config()

        x = d0 if i == 1 else d

        l_type = type(layer)

        if l_type is Conv2D:
            cnt += 2*dil_rate

            d = Conv2D(config["filters"], dilation_rate=dil_rate, kernel_size=config["kernel_size"], activation=config["activation"])(x)

            if not config["strides"] == (1, 1):
                dil_rate *= config["strides"][0]

        elif l_type is Swish:
            d = Swish(config["beta"])(d)
        else:
            d = l_type()(d)

    dis = Model(d0, d, name="new_dis")
    dis.set_weights(orig_dis.get_weights())

    return dis, cnt


########################################################################################################################


def tv_seg_primal_dual(f, maxits):
    def dxm(u):
        M, _ = u.shape
        temp = u[:, :-1]
        return np.hstack((temp, np.zeros((M, 1)))) - np.hstack((np.zeros((M, 1)), temp))


    def dxp(u):
        return np.hstack((u[:, 1:], u[:, -1:])) - u


    def dym(u):
        _, N = u.shape
        temp = u[:-1, :]
        return np.vstack((temp, np.zeros((1, N)))) - np.vstack((np.zeros((1, N)), temp))


    def dyp(u):
        return np.vstack((u[1:, :], u[-1:, :])) - u

    M, N = f.shape

    u = np.zeros((M, N))
    u_ = np.zeros((M, N))
    p = np.zeros((M, N, 2))

    L2 = 8
    tau = 1 / np.sqrt(L2)
    sigma = 1 / tau / L2

    for k in range(maxits):
        u_x = dxp(u_)
        u_y = dyp(u_)

        p[:, :, 0] = p[:, :, 0] + sigma * u_x
        p[:, :, 1] = p[:, :, 1] + sigma * u_y

        reprojection = np.clip(np.sqrt(p[:, :, 0]**2 + p[:, :, 1]**2), 1.0, None)

        p[:, :, 0] = p[:, :, 0] / reprojection
        p[:, :, 1] = p[:, :, 1] / reprojection

        div = dxm(p[:, :, 0]) + dym(p[:, :, 1])
        u_ = u
        u = np.clip(u + tau * (div - f), 0.0, 1.0)

        u_ = 2 * u - u_

    return u

########################################################################################################################


if args.images is None:
    numImages = len(images)
    img_list = np.arange(numImages)
else:
    img_list = args.images
    numImages = len(img_list)

_, c1, c2 = images.shape
new_shape = (numImages, c1, c2)
final = np.zeros(new_shape, dtype=bool)

######################################
if debug_mode:
    all_pred = np.zeros(new_shape)
    all_reg = np.zeros(new_shape)
######################################

factor = 1
num_rows, _ = images[0].shape

predictor, pad_size = build_predictor(dis)

for i, img_nr in enumerate(img_list):
    print("pred: {} {}".format(i, time.ctime(time.time())))

    pred_y = predict_image(images[img_nr], predictor, pad_size, factor)
    reg_y = tv_seg_primal_dual((pred_y - bias)*lamb, num_tv_it)
    thr_y = (reg_y[:, :] < 0.5)
    final[- 1 - i] = thr_y

    ######################################
    if debug_mode:
        all_pred[i] = pred_y
        all_reg[i] = reg_y

        np.save("{}_all_pred".format(ID), all_pred)
        np.save("{}_all_reg".format(ID), all_reg)
        np.save("{}_final".format(ID), final)
    ######################################

imsave(file_out, final, photometric='minisblack')
