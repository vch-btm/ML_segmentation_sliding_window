from __future__ import print_function, division
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import keras
from keras.models import load_model, Input, Model
from keras.layers import Conv2D
from keras.optimizers import Optimizer
import tensorflow as tf
import numpy as np
import time
from tifffile import imsave
import mrcfile
import argparse
import pyopencl as cl
import pyopencl.array as array

from keras.layers import Layer
from keras import backend as K


#########################################################

class Swish(Layer):
    def __init__(self, beta, **kwargs):
        super(Swish, self).__init__(**kwargs)
        self.beta = K.cast_to_floatx(beta)

    def call(self, inputs):
        return K.sigmoid(self.beta * inputs) * inputs

    def get_config(self):
        config = {'beta': float(self.beta)}
        base_config = super(Swish, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class Mish(Layer):
    # mish(x) = x*tanh(softplus(x)) = x*tanh(ln(1+e^x))

    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))

    def get_config(self):
        base_config = super(Mish, self).get_config()
        return dict(list(base_config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class RAdam2(Optimizer):
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0., weight_decay=0., amsgrad=False, total_steps=0, warmup_proportion=0.1, min_lr=0., **kwargs):
        learning_rate = kwargs.pop('lr', learning_rate)
        super(RAdam2, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.learning_rate = K.variable(learning_rate, name='learning_rate')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
            self.weight_decay = K.variable(weight_decay, name='weight_decay')
            self.total_steps = K.variable(total_steps, name='total_steps')
            self.warmup_proportion = K.variable(warmup_proportion, name='warmup_proportion')
            self.min_lr = K.variable(min_lr, name='min_lr')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.initial_weight_decay = weight_decay
        self.initial_total_steps = total_steps
        self.amsgrad = amsgrad

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr

        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations, K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1

        if self.initial_total_steps > 0:
            warmup_steps = self.total_steps * self.warmup_proportion
            decay_steps = K.maximum(self.total_steps - warmup_steps, 1)
            decay_rate = (self.min_lr - lr) / decay_steps
            lr = K.switch(
                t <= warmup_steps,
                lr * (t / warmup_steps),
                lr + decay_rate * K.minimum(t - warmup_steps, decay_steps),
            )

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p), name='m_' + str(i)) for (i, p) in enumerate(params)]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p), name='v_' + str(i)) for (i, p) in enumerate(params)]

        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p), name='vhat_' + str(i)) for (i, p) in enumerate(params)]
        else:
            vhats = [K.zeros(1, name='vhat_' + str(i)) for i in range(len(params))]

        self.weights = [self.iterations] + ms + vs + vhats

        beta_1_t = K.pow(self.beta_1, t)
        beta_2_t = K.pow(self.beta_2, t)

        sma_inf = 2.0 / (1.0 - self.beta_2) - 1.0
        sma_t = sma_inf - 2.0 * t * beta_2_t / (1.0 - beta_2_t)

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)

            m_corr_t = m_t / (1.0 - beta_1_t)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                v_corr_t = K.sqrt(vhat_t / (1.0 - beta_2_t))
                self.updates.append(K.update(vhat, vhat_t))
            else:
                v_corr_t = K.sqrt(v_t / (1.0 - beta_2_t))

            r_t = K.sqrt((sma_t - 4.0) / (sma_inf - 4.0) *
                         (sma_t - 2.0) / (sma_inf - 2.0) *
                         sma_inf / sma_t)

            p_t = K.switch(sma_t >= 5, r_t * m_corr_t / (v_corr_t + self.epsilon), m_corr_t)

            if self.initial_weight_decay > 0:
                p_t += self.weight_decay * p

            p_t = p - lr * p_t

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    @property
    def lr(self):
        return self.learning_rate

    @lr.setter
    def lr(self, learning_rate):
        self.learning_rate = learning_rate

    def get_config(self):
        config = {
            'learning_rate': float(K.get_value(self.learning_rate)),
            'beta_1': float(K.get_value(self.beta_1)),
            'beta_2': float(K.get_value(self.beta_2)),
            'decay': float(K.get_value(self.decay)),
            'weight_decay': float(K.get_value(self.weight_decay)),
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad,
            'total_steps': float(K.get_value(self.total_steps)),
            'warmup_proportion': float(K.get_value(self.warmup_proportion)),
            'min_lr': float(K.get_value(self.min_lr)),
        }
        base_config = super(RAdam2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


#########################################################


class Program(object):
    def __init__(self, ctx, code):
        self._cl_prg = cl.Program(ctx, code)
        self._cl_prg.build()
        self._cl_kernels = self._cl_prg.all_kernels()
        for kernel in self._cl_kernels:
            self.__dict__[kernel.function_name] = kernel


ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

prg = Program(ctx, """
__kernel void dxp(__global float *dest, __global float *src) {
  size_t Nx = get_global_size(0);
  size_t x = get_global_id(0), y = get_global_id(1);
  size_t i = y*Nx + x;

  if (x == Nx-1) dest[i] = 0.0f;
  else dest[i] = src[i+1] - src[i];
}

__kernel void dyp(__global float *dest, __global float *src) {
  size_t Nx = get_global_size(0), Ny = get_global_size(1);
  size_t x = get_global_id(0), y = get_global_id(1);
  size_t i = y*Nx + x;

  if (y == Ny-1) dest[i] = 0.0f;
  else dest[i] = src[i+Nx] - src[i];
}

__kernel void dxm(__global float *dest, __global float *src) {
  size_t Nx = get_global_size(0);
  size_t x = get_global_id(0), y = get_global_id(1);
  size_t i = y*Nx + x;

  float val = 0.0f;
  if (x < Nx-1) val += src[i];
  if (x > 0) val -= src[i-1];
  dest[i] = val;
}

__kernel void dym(__global float *dest, __global float *src) {
  size_t Nx = get_global_size(0), Ny = get_global_size(1);
  size_t x = get_global_id(0), y = get_global_id(1);
  size_t i = y*Nx + x;

  float val = 0.0f;
  if (y < Ny-1) val += src[i];
  if (y > 0) val -= src[i-Nx];
  dest[i] = val;
}

__kernel void grad(__global float2 *dest, __global float *src) {
  size_t Nx = get_global_size(0), Ny = get_global_size(1);
  size_t x = get_global_id(0), y = get_global_id(1);
  size_t i = y*Nx + x;

  float2 val = src[i];
  if (x < Nx-1) val.s0 -= src[i+1];  else val.s0 = 0.0f;
  if (y < Ny-1) val.s1 -= src[i+Nx]; else val.s1 = 0.0f;
  dest[i] = val;
}

__kernel void div(__global float *dest, __global float2 *src) {
  size_t Nx = get_global_size(0), Ny = get_global_size(1);
  size_t x = get_global_id(0), y = get_global_id(1);
  size_t i = y*Nx + x;

  float2 val = src[i];
  if (x == Nx-1) val.s0 = 0.0f;
  if (x > 0) val.s0 -= src[i-1].s0;
  if (y == Ny-1) val.s1 = 0.0f;
  if (y > 0) val.s1 -= src[i-Nx].s1;
  dest[i] = val.s0 + val.s1;
}

__kernel void reproject(__global float2 *dest, __global float2 *src, const float alphainv) {
  size_t Nx = get_global_size(0);
  size_t x = get_global_id(0), y = get_global_id(1);
  size_t i = Nx*y + x;

  float2 val = src[i];
  float fac = hypot(val.s0, val.s1)*alphainv;
  if (fac > 1.0f) dest[i] = val/fac;
  else dest[i] = val;
}

__kernel void update_p(__global float2 *p, __global float *u, const float sigma) {
  size_t Nx = get_global_size(0), Ny = get_global_size(1);
  size_t x = get_global_id(0), y = get_global_id(1);
  size_t i = Nx*y + x;

  // gradient 
  float2 val = -u[i];
  if (x < Nx-1) val.s0 += u[i+1];  else val.s0 = 0.0f;
  if (y < Ny-1) val.s1 += u[i+Nx]; else val.s1 = 0.0f;

  // step
  val = p[i] + sigma*val;

  // reproject
  float fac = hypot(val.s0, val.s1);
  if (fac > 1.0f) p[i] = val/fac; else p[i] = val;
}

__kernel void update_u(__global float *u_, __global float *u, __global float2 *p, __global float *f, const float tau) {
  size_t Nx = get_global_size(0), Ny = get_global_size(1);
  size_t x = get_global_id(0), y = get_global_id(1);
  size_t i = y*Nx + x;

  // divergence
  float2 val = p[i];
  if (x == Nx-1) val.s0 = 0.0f;
  if (x > 0) val.s0 -= p[i-1].s0;
  if (y == Ny-1) val.s1 = 0.0f;
  if (y > 0) val.s1 -= p[i-Nx].s1;

  // linear step
  float v;
  v = u[i] + tau*(val.s0 + val.s1 - f[i]);
  if (v > 1.0f) v = 1.0f;
  if (v < 0.0f) v = 0.0f;
  u_[i] = v;
}

""")


# prg = { k.function_name : k for k in prg.all_kernels() }

def dxp(dest, src):
    return prg.dxp(src.queue, src.shape, None, dest.data, src.data)


def dyp(dest, src):
    return prg.dyp(src.queue, src.shape, None, dest.data, src.data)


def dxm(dest, src):
    return prg.dxm(src.queue, src.shape, None, dest.data, src.data)


def dym(dest, src):
    return prg.dym(src.queue, src.shape, None, dest.data, src.data)


def update_p(p, u, sigma):
    return prg.update_p(p.queue, u.shape, None, p.data, u.data, np.float32(sigma))


def update_u(u_, u, p, f, tau):
    return prg.update_u(u.queue, u.shape, None, u_.data, u.data, p.data, f.data, np.float32(tau))


update_u_ = cl.elementwise.ElementwiseKernel(ctx, 'float *u, float *u_', 'u[i] = 2.0f*u_[i] - u[i]')

########################################################################################################################

ID = "{}".format(time.strftime("%Y%m%d_%H%M"))

parser = argparse.ArgumentParser(description='predict images')
parser.add_argument('-file_in', metavar='IN', type=str, default="labeled_data/380-1_reconstructed.mrc", help="name of input file")
parser.add_argument('-file_out', metavar='OUT', type=str, default="", help="name of output file")
parser.add_argument("-dis", metavar="DIS", type=str, default="", help="name of classifier")
parser.add_argument("-nit", metavar="NIT", type=int, default=50, help="number of iterations (regularization)")
parser.add_argument("-b", metavar="bias", type=float, default=0.5, help="bias (regularization)")
parser.add_argument("-l", metavar="lambda", type=float, default=0.85, help="lambda (regularization)")
parser.add_argument("-t", metavar="threshold", type=float, default=0.85, help="value for thresholding")
parser.add_argument("-f", metavar="factor", type=float, default=1, help="factor for prediction")
parser.add_argument("-debug", metavar="debug", type=bool, default=False, help="activate debug mode")
parser.add_argument('-images', metavar='N', type=int, nargs='+', help='images to process (starting with 0)')

args = parser.parse_args()

#########################################################

file_in = args.file_in
file_out = args.file_out

if file_out == "":
    split_txt = args.dis.split("_")
    file_out = "tif_{}_{}_{}_{}".format(split_txt[0], split_txt[1], split_txt[2], ID)

with keras.utils.CustomObjectScope({'Swish': Swish, 'Mish': Mish, 'RAdam2': RAdam2}):
    dis = load_model(args.dis)
bias = args.b
lamb = args.l
thr = args.t
factor = args.f
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

# @profile
def predict_image(image, predictor, pad_size, factor=1):
    (num_rows, num_cols) = image.shape

    fake_y = np.zeros(image.shape)

    image = np.pad(image, pad_size // 2, 'symmetric')

    ############################################################################

    n_batches = factor * factor
    rows_part = num_rows // factor
    cols_part = num_cols // factor

    w_size = rows_part + pad_size

    x_data = np.zeros((n_batches, w_size, w_size, 1))

    for j in range(n_batches):
        coord_x, coord_y = divmod(j, factor)

        x_data[j] = image[coord_x * rows_part:(coord_x + 1) * rows_part + pad_size,
                    coord_y * cols_part:(coord_y + 1) * cols_part + pad_size].reshape(w_size, w_size, 1)

    ############################################################################

    pred = predictor.predict(x_data)

    for i in range(factor):
        for j in range(factor):
            fake_y[i * rows_part:(i + 1) * rows_part, j * cols_part:(j + 1) * cols_part] = pred[i * factor + j].reshape((rows_part, rows_part))

    return fake_y


########################################################################################################################


def build_predictor(orig_dis):
    num_layers = len(orig_dis.layers)

    dil_rate = 1
    cnt = 0

    d0 = Input(shape=(None, None, 1))

    for i in range(1, num_layers):
        layer = orig_dis.layers[i]
        conf = layer.get_config()

        x = d0 if i == 1 else x

        l_type = type(layer)

        if l_type is Conv2D:
            cnt += (conf["kernel_size"][0] - 1) * dil_rate

            x = Conv2D(conf["filters"], dilation_rate=dil_rate, kernel_size=conf["kernel_size"], activation=conf["activation"])(x)

            if not conf["strides"] == (1, 1):
                dil_rate *= conf["strides"][0]

        elif l_type is Mish:
            x = Mish()(x)
        else:
            x = l_type()(x)

    dis = Model(d0, x, name="new_dis")
    dis.set_weights(orig_dis.get_weights())

    return dis, cnt


########################################################################################################################


def tv_seg_primal_dual(f0, maxits):
    f = array.to_device(queue, f0.astype(np.float32, order='F'))
    u = array.zeros_like(f)
    u_ = array.zeros_like(f)
    p = array.zeros(queue, (2, u.shape[0], u.shape[1]), dtype=np.float32, order='F')

    L2 = 8
    tau = 1.0 / np.sqrt(L2)
    sigma = 1.0 / tau / L2

    for k in range(maxits):
        update_p(p, u_, sigma)
        update_u(u_, u, p, f, tau)
        update_u_(u, u_)
        (u, u_) = (u_, u)

    return u.get()


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

# factor = 8
num_rows, _ = images[0].shape

predictor, pad_size = build_predictor(dis)

t0 = time.time()

for i, img_nr in enumerate(img_list):
    print("pred: {} {}".format(i, time.ctime(time.time())))
    img = images[i]
    img = np.clip((img - img.mean()) / (img.std() + 1e-8), -3, 3)
    pred_y = predict_image(img, predictor, pad_size, factor)
    reg_y = tv_seg_primal_dual((pred_y - bias) * lamb, num_tv_it)
    thr_y = (reg_y[:, :] < thr)
    final[- 1 - i] = thr_y

    ######################################
    if debug_mode:
        all_pred[i] = pred_y
        all_reg[i] = reg_y

        np.save("{}_all_pred".format(ID), all_pred)
        np.save("{}_all_reg".format(ID), all_reg)
        np.save("{}_final".format(ID), final)

    ######################################

t1 = time.time()

imsave(file_out, final, photometric='minisblack')

print("finished after {} seconds".format(t1 - t0))
