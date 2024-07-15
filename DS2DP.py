#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: yuchun   time: 2020/7/10
from collections import namedtuple
import scipy.io as scio
from com_psnr import quality
from net import *
from net.fcn import fcn
from net.losses import *
from net.noise import *
from utils.image_io import *
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
Result = namedtuple("Result", ['recon', 'psnr', 'guass', 'salt', 'AB', 'c'])

class Dehaze(object):
    def __init__(self, path, filename, image_name, image, image_clean, num_iter=999999999000, plot_during_training=True,
                 show_every=500,
                 use_deep_channel_prior=True,
                 gt_ambient=None, clip=True):
        self.start_rank = 1
        self.max_rank = 7
        self.now_rank = self.start_rank
        self.path = path
        self.filename = filename
        self.image_name = image_name
        self.image = image
        self.image_clean = image_clean
        self.num_iter = num_iter
        self.plot_during_training = plot_during_training
        self.show_every = show_every
        self.use_deep_channel_prior = use_deep_channel_prior
        self.gt_ambient = gt_ambient
        self.ambient_net = None
        self.image_net = None
        self.mask_net = None
        self.ambient_val = None
        self.mse_loss = None
        self.learning_rate = 0.001
        self.parameters = None
        self.current_result = None
        self.input_depth = 1
        self.output_depth = 1
        self.exp_weight = 0.98
        self.clip = clip
        self.blur_loss = None
        self.best_result = None
        self.best_result_av = None
        self.image_net_inputs = None
        self.mask_net_inputs = None
        self.image_out = None
        self.mask_out = None
        self.done = False
        self.ambient_out = None
        self.total_loss = None
        self.post = None
        self._init_all()
        self.out_avg = 0
        self.save_every = 1000
        self.o = torch.zeros((self.image_clean.shape[0] * self.image_clean.shape[1], self.image_clean.shape[2])).type(
            torch.cuda.FloatTensor)
        self.s = torch.zeros((self.image_clean.shape[0] * self.image_clean.shape[1], self.image_clean.shape[2])).type(
            torch.cuda.FloatTensor)
        self.lanmda = 0.01
        self.previous = np.zeros(self.image_clean.shape)

    def update(self):
        self.s = torch.max(self.o, self.image_torch - self.image_com - self.lanmda) + torch.min(self.o,
                                                                        self.image_torch - self.image_com + self.lanmda)
    def add_rank(self):
        pad = 'reflection'
        data_type = torch.cuda.FloatTensor
        net = skip(self.input_depth, self.output_depth,  num_channels_down = [16, 32, 64, 128, 128, 128],
                           num_channels_up =   [16, 32, 64, 128, 128, 128],
                           num_channels_skip = [0, 0, 4, 4, 4, 4],
                           filter_size_down = [7, 7, 5, 5, 3, 3], filter_size_up = [7, 7, 5, 5, 3, 3],
                           upsample_mode='bilinear', downsample_mode='avg',
                           need_sigmoid=True, pad=pad, act_fun='LeakyReLU').type(data_type)

        net.load_state_dict(torch.load('image_net.pth'))
        self.image_net.append(net)
        self.parameters = self.parameters + [p for p in net.parameters()]
        net = fcn(self.image_clean.shape[2], self.image_clean.shape[2], num_hidden=[128, 256, 256, 128]).type(data_type)
        net.load_state_dict(torch.load('mask_net.pth'))
        self.mask_net.append(net)
        self.parameters = self.parameters + [p for p in net.parameters()]

        self.now_rank = self.now_rank + 1

        self.optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)


    def _init_images(self):
        self.original_image = self.image.copy()
        image = self.image
        self.image_torch = np_to_torch(image).type(torch.cuda.FloatTensor)
        self.image_torch = self.image_torch.squeeze(0)

    def _init_nets(self):
        pad = 'reflection'
        data_type = torch.cuda.FloatTensor
        self.image_net = []
        self.parameters = []
        for i in range(self.start_rank):
            net = skip(self.input_depth, self.output_depth,  num_channels_down = [16, 32, 64, 128, 128, 128],
                           num_channels_up = [16, 32, 64, 128, 128, 128],
                           num_channels_skip = [0, 0, 4, 4, 4, 4],
                           filter_size_down = [7, 7, 5, 5, 3, 3], filter_size_up = [7, 7, 5, 5, 3, 3],
                           upsample_mode='bilinear', downsample_mode='avg',
                           need_sigmoid=True, pad=pad, act_fun='LeakyReLU').type(data_type)
            self.parameters = [p for p in net.parameters()] + self.parameters
            self.image_net.append(net)
        self.mask_net = []
        for i in range(self.start_rank):
            net = fcn(self.image_clean.shape[2], self.image_clean.shape[2], num_hidden=[128, 256, 256, 128]).type(data_type)
            self.parameters = self.parameters + [p for p in net.parameters()]
            self.mask_net.append(net)

    def generate(self, a, dimension):
        if dimension == 2:
            noisy = np.zeros((a, a))
            bar = np.linspace(0, 1, 2 * a - 1)
            w, l = np.indices(noisy.shape)
            for k in range(2 * a - 1):
                noisy[w == l - a + 1 + k] = bar[k]
            noisy = noisy[np.newaxis, :]
        elif dimension == 1:
            noisy = np.linspace(0, 1, a)
        return noisy


    def _init_parameters(self):
        return 0

    def _init_loss(self):
        data_type = torch.cuda.FloatTensor
        self.mse_loss = torch.nn.MSELoss().type(data_type)
        self.sp_loss = SPLoss().type(data_type)

    def _init_inputs(self):
        # original_nois: 12*1*145*145
        original_noise = torch_to_np(get_noise1(1, 'noise', (self.input_depth, self.image_clean.shape[0], self.image_clean.shape[1]), noise_type='u',
                                                                     var=10/10.).type(torch.cuda.FloatTensor).detach())
        self.image_net_inputs = np_to_torch(original_noise).type(torch.cuda.FloatTensor).detach()[0, :, :, :]


        original_noise = torch_to_np(get_noise2(1, 'noise', self.image.shape[1], noise_type='u', var=10/ 10.).type(torch.cuda.FloatTensor).detach())
        self.mask_net_inputs = np_to_torch(original_noise).type(torch.cuda.FloatTensor).detach()[0, :, :, :]
        self.mask_net_inputs = self.mask_net_inputs

    def _init_all(self):
        self._init_images()
        self._init_nets()
        self._init_inputs()
        self._init_parameters()
        self._init_loss()

    def optimize(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        self.optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        for j in range(self.num_iter + 1):
            self.optimizer.zero_grad()
            self._optimization_closure(j)
            self._obtain_current_result()
            self._plot_closure(j)
            self.optimizer.step()

    def _optimization_closure(self, step):
        if (step % 3000 == 0) & (self.now_rank < self.max_rank):
            torch.save(self.image_net[-1].state_dict(), 'image_net.pth')
            torch.save(self.mask_net[-1].state_dict(), 'mask_net.pth')
            self.add_rank()
            self.add_rank()
            self.add_rank()

        m = 0
        M = self.image_net_inputs
        out = self.image_net[0](M)
        for i in range(1, self.now_rank):
            out = torch.cat((out, self.image_net[i](M)), 0)
        out = out[:, :, :self.image_clean.shape[0], :self.image_clean.shape[1]]
        self.image_out = out[m, :, :, :].squeeze().reshape((-1, 1))
        for m in range(1, self.now_rank):
            self.image_out = torch.cat((self.image_out, out[m, :, :, :].squeeze().reshape((-1, 1))), 1)
        self.image_out_np = torch_to_np(self.image_out)

        M = self.mask_net_inputs
        out = self.mask_net[0](M)
        for i in range(1, self.now_rank):
            out = torch.cat((out, self.mask_net[i](M)), 0)
        self.mask_out = out.squeeze(1)
        self.mask_out_np = torch_to_np(self.mask_out)
        self.image_com = self.image_out.mm(self.mask_out)
        self.image_com_np = np.matmul(self.image_out_np, self.mask_out_np)
        self.image_com_np = np.reshape(self.image_com_np, self.image_clean.shape, order='F')
        self.out_avg = self.out_avg * self.exp_weight + self.image_com_np * (1 - self.exp_weight)

        if step > 200:
            self.update()
        self.guass = torch_to_np(self.image_torch - self.image_com - self.s)

        self.total_loss = self.mse_loss(self.image_com + self.s, self.image_torch) + 2 * self.lanmda * self.sp_loss(self.s)
        self.total_loss.backward(retain_graph=True)
        self.res = np.sqrt(np.sum(np.square(self.out_avg - self.previous)) / np.sum(np.square(self.previous)))
        self.previous = self.out_avg

    def _obtain_current_result(self):
        self.psnr = quality(self.image_clean, self.image_com_np.astype(np.float64))
        self.psnr_av = quality(self.image_clean, self.out_avg.astype(np.float64))
        self.current_result = Result(recon=self.image_com_np,  psnr=self.psnr, guass=self.guass, salt=torch_to_np(self.s), AB=self.image_out_np, c=self.mask_out_np)
        self.current_result_av = Result(recon=self.out_avg,  psnr=self.psnr_av, guass=self.guass, salt=torch_to_np(self.s), AB=self.image_out_np, c=self.mask_out_np)
        if self.best_result is None or self.best_result.psnr < self.current_result.psnr:
            self.best_result = self.current_result
        if self.best_result_av is None or self.best_result_av.psnr < self.current_result_av.psnr:
            self.best_result_av = self.current_result_av

    def _plot_closure(self, step):
        print('Iteration %05d  tol_loss %f    current_psnr: %f  max_psnr %f  current_psnr_av: %f max_psnr_av: %f rank: %d   res: %f'   % (step, self.total_loss.item(),
                                                                                self.current_result.psnr, self.best_result.psnr,
                                                                                self.current_result_av.psnr, self.best_result_av.psnr, self.now_rank, self.res ), '\r')


def dehaze(path, filename, result, image_name, image, image_clean, num_iter=10000, plot_during_training=True,
           show_every=500,
           use_deep_channel_prior=True,
           gt_ambient=None):
    dh = Dehaze(path, filename, image_name + "_0", image, image_clean, num_iter, plot_during_training, show_every, use_deep_channel_prior,
                gt_ambient, clip=True)
    dh.optimize()



if __name__ == "__main__":
    result = []
    dataname = {
        0: "wdc_h",
        1: "pavia",
        2: "paviac",
    }
    for case in range(5, 6):
        for num in range(0, 1):
            print("case %d num %d" % (case, num))
            path = "images//Case{}".format(case)
            filename = dataname.get(num)
            mat = scipy.io.loadmat(os.path.join(path, filename + ".mat"))
            image_clean = mat["img_clean"]
            image_noisy = mat["img_noisy"]
            image_noisy = np.reshape(image_noisy, (image_noisy.shape[0] * image_noisy.shape[1], image_noisy.shape[2]), order="F")
            result = dehaze(path, filename, result, "hs_noisy", image_noisy, image_clean, use_deep_channel_prior=True, gt_ambient=np.array([0.5600084 , 0.64564645, 0.72515032]))
