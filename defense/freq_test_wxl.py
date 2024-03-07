# MIT License

# Copyright (c) 2017 Brandon Tran and Jerry Li

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''
This file is modified from spectral.py, in order to test frequency domain defense.
'''

from torch import Tensor, torch
from torch.fft import fft2, ifft2, fftshift, ifftshift
from torchvision import transforms
from torchvision.utils import _log_api_usage_once
import time
import logging
import yaml
from pprint import pformat
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.append('../')
sys.path.append(os.getcwd())

from defense.base import defense
from utils.aggregate_block.train_settings_generate import argparser_criterion, argparser_opt_scheduler
from utils.trainer_cls import PureCleanModelTrainer
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.log_assist import get_git_info
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.save_load_attack import load_attack_result, save_defense_result
from utils.trainer_cls import Metric_Aggregator

# the low frequency substitution module

class low_freq_substitution:
    # now use a fix image
    def __init__(self, input_height, input_width, low_freq_image, alpha, beta=1) -> None:
        _log_api_usage_once(self)
        # the shape of image is [C.H.W]
        # assert if not match the shape
        assert low_freq_image.shape[0] == 3 and low_freq_image.shape[
            1] == input_height and low_freq_image.shape[2] == input_width, 'the shape of low_freq_image should be [3, input_height, input_width]'
        self.alpha = alpha
        self.beta = beta
        self.input_height = input_height
        self.input_width = input_width
        # prepare low frequency mask and low frequency fft
        # shape of low_freq_image_fft is [3, input_height, input_width]
        low_freq_image_fft = fft2(low_freq_image, dim=(-2, -1))
        low_freq_image_fft = fftshift(low_freq_image_fft, dim=(-2, -1))
        low_freq_image_fft = torch.abs(low_freq_image_fft)
        self.low_freq_image_fft = low_freq_image_fft
        center = ((input_height-1)/2, (input_width-1)/2)
        max_radius = min(
            center[0], center[1], input_height-center[0], input_width-center[1])
        radius = max_radius*alpha
        self.mask = torch.zeros(input_height, input_width)
        for i in range(input_height):
            for j in range(input_width):
                if (i-center[0])**2 + (j-center[1])**2 <= radius**2:
                    self.mask[i][j] = self.beta

    # replace the low frequency part of the image with the low frequency part of a random image
    def forward(self, tensor: Tensor) -> Tensor:
        # get the amplitude and phase of the input image
        tensor_fft = fft2(tensor, dim=(-2, -1))
        tensor_fft = fftshift(tensor_fft, dim=(-2, -1))
        tensor_amplitude = torch.abs(tensor_fft)
        tensor_phase = torch.angle(tensor_fft)
        # replace low frequency part with self.low_freq_image_fft and mask
        tensor_amplitude = self.mask * self.low_freq_image_fft + \
            (1-self.mask) * tensor_amplitude
        # get the new image tensor
        tensor_fft = torch.polar(tensor_amplitude, tensor_phase)
        tensor_fft = ifftshift(tensor_fft, dim=(-2, -1))
        tensor = ifft2(tensor_fft, dim=(-2, -1))
        tensor = torch.abs(tensor)
        return tensor

    def __call__(self, tensor: Tensor) -> Tensor:
        return self.forward(tensor)

    def __repr__(self) -> str:
        return f"low_freq_substitution(alpha={self.alpha}, beta={self.beta})"


class spectral(defense):

    def __init__(self, args):
        with open(args.yaml_path, 'r') as f:
            defaults = yaml.safe_load(f)

        defaults.update(
            {k: v for k, v in args.__dict__.items() if v is not None})

        args.__dict__ = defaults

        args.terminal_info = sys.argv

        args.num_classes = get_num_classes(args.dataset)
        args.input_height, args.input_width, args.input_channel = get_input_shape(
            args.dataset)
        args.img_size = (args.input_height, args.input_width,
                         args.input_channel)
        args.dataset_path = f"{args.dataset_path}/{args.dataset}"

        self.args = args

        if 'result_file' in args.__dict__:
            if args.result_file is not None:
                self.set_result(args.result_file)

    def add_arguments(parser):
        parser.add_argument('--device', type=str, help='cuda, cpu')

        parser.add_argument('--checkpoint_load', type=str,
                            help='the location of load model')
        parser.add_argument('--checkpoint_save', type=str,
                            help='the location of checkpoint where model is saved')
        parser.add_argument('--log', type=str, help='the location of log')
        parser.add_argument("--dataset_path", type=str,
                            help='the location of data')
        parser.add_argument('--dataset', type=str,
                            help='mnist, cifar10, cifar100, gtrsb, tiny')
        parser.add_argument('--result_file', type=str,
                            help='the location of result')

        parser.add_argument('--epochs', type=int)
        parser.add_argument('--batch_size', type=int)
        parser.add_argument("--num_workers", type=float)
        parser.add_argument('--lr', type=float)
        parser.add_argument('--lr_scheduler', type=str,
                            help='the scheduler of lr')
        parser.add_argument('--steplr_stepsize', type=int)
        parser.add_argument('--steplr_gamma', type=float)
        parser.add_argument('--steplr_milestones', type=list)
        parser.add_argument('--model', type=str, help='resnet18')

        parser.add_argument('--client_optimizer', type=int)
        parser.add_argument('--sgd_momentum', type=float)
        parser.add_argument('--wd', type=float, help='weight decay of sgd')
        parser.add_argument('--frequency_save', type=int,
                            help=' frequency_save, 0 is never')

        parser.add_argument('--random_seed', type=int, help='random seed')

        # parameter related to the frequency domain processing
        parser.add_argument('--alpha', type=float, default=0.01,
                             help='the percentage of the low frequency part of the image')
        parser.add_argument('--beta', type=float, default=1,
                             help='mask amplification factor')
        # parser.add_argument('--yaml_path', type=str, default="./config/defense/spectral/config.yaml", help='the path of yaml')

        # # set the parameter for the spectral defense
        # parser.add_argument('--percentile', type=float)
        # parser.add_argument('--target_label', type=int)

    def set_result(self, result_file):
        attack_file = './record/' + result_file
        save_path = './record/' + result_file + '/defense/freq_test_wxl/'
        if not (os.path.exists(save_path)):
            os.makedirs(save_path)
        # assert(os.path.exists(save_path))
        self.args.save_path = save_path
        if self.args.checkpoint_save is None:
            self.args.checkpoint_save = save_path + 'checkpoint/'
            if not (os.path.exists(self.args.checkpoint_save)):
                os.makedirs(self.args.checkpoint_save)
        if self.args.log is None:
            self.args.log = save_path + 'log/'
            if not (os.path.exists(self.args.log)):
                os.makedirs(self.args.log)
        self.result = load_attack_result(attack_file + '/attack_result.pt')

    def set_trainer(self, model):
        self.trainer = PureCleanModelTrainer(
            model,
        )

    def set_logger(self):
        args = self.args
        logFormatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
        )
        logger = logging.getLogger()

        fileHandler = logging.FileHandler(
            args.log + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        logger.addHandler(consoleHandler)

        logger.setLevel(logging.INFO)
        logging.info(pformat(args.__dict__))

        try:
            logging.info(pformat(get_git_info()))
        except:
            logging.info('Getting git info fails.')

    def set_devices(self):
        self.device = self.args.device

    def mitigation(self):
        # set the device and random seed
        self.set_devices()
        fix_random(self.args.random_seed)

        # prepare the model and device
        model = generate_cls_model(self.args.model, self.args.num_classes)
        model.load_state_dict(self.result['model'])
        if "," in self.device:
            model = torch.nn.DataParallel(
                model,
                # eg. "cuda:2,3,7" -> [2,3,7]
                device_ids=[int(i) for i in self.args.device[5:].split(",")]
            )
            self.args.device = f'cuda:{model.device_ids[0]}'
            model.to(self.args.device)
        else:
            model.to(self.args.device)
        logging.info(f'Using device: {self.args.device}')

        # Setting up the data and the model
        train_trans = get_transform(
            self.args.dataset, *([self.args.input_height, self.args.input_width]), train=True)
        test_trans = get_transform(
            self.args.dataset, *([self.args.input_height, self.args.input_width]), train=False)

        # set the clean trainning dataset and its transform
        data_clean_trainset = self.result['clean_train']
        data_clean_trainset.wrap_img_transform = train_trans
        data_clean_loader = torch.utils.data.DataLoader(data_clean_trainset, batch_size=self.args.batch_size,
                                                        num_workers=self.args.num_workers, drop_last=False, shuffle=True, pin_memory=args.pin_memory)

        # set bd dataset and its transform
        data_bd_testset = self.result['bd_test']
        data_bd_testset.wrap_img_transform = test_trans

        # define a new transform for the frequency domain processing
        # original transform: 0.resize 1.random crop 2.totensor 3.normalize
        # new transform: 0.resize 1.random crop 2.totensor 3.frequancy domain processing 4.normalize
        # change the transform of the bd dataset
        # random choose a low frequency image
        length = len(data_bd_testset)
        index = np.random.randint(0, length)
        low_freq_image = data_bd_testset[index][0]
        # low frequency image transform
        low_freq_image_transform = transforms.Compose([
            transforms.Resize((self.args.input_height, self.args.input_width)),
            transforms.ToTensor()
        ])
        low_freq_image = low_freq_image_transform(low_freq_image)
        # prepare the low frequency substitution module
        low_freq_sub = low_freq_substitution(
            self.args.input_height, self.args.input_width, low_freq_image, self.args.alpha, self.args.beta)
        # add the low frequency substitution module to the transform
        logging.info("changing the transform of the bd dataset")
        logging.info("orignal transforms",test_trans)
        temp = test_trans[-1]
        test_trans[-1] = low_freq_sub
        test_trans.append(temp)
        logging.info("new transforms",test_trans)

        # add transform to the bd dataset
        data_bd_testset.wrap_img_transform = test_trans


        data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=self.args.batch_size,
                                                     num_workers=self.args.num_workers, drop_last=False, shuffle=False, pin_memory=args.pin_memory)

        # set clean dataset and its transform
        data_clean_testset = self.result['clean_test']
        data_clean_testset.wrap_img_transform = test_trans
        data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=self.args.batch_size,
                                                        num_workers=self.args.num_workers, drop_last=False, shuffle=False, pin_memory=args.pin_memory)

        # set the dataloader dict
        test_dataloader_dict = {}
        test_dataloader_dict['clean_test_dataloader'] = data_clean_loader
        test_dataloader_dict['bd_test_dataloader'] = data_bd_loader

        # set train dataloader
        train_dataloader = data_clean_loader

        # in order to logging the result together
        agg = Metric_Aggregator()

        # put the model into trainer handle
        self.set_trainer(model)

        # get the criterion, optimizer and scheduler
        # default: cross entropy loss
        criterion = argparser_criterion(args)
        # optimizer and scheduler are set by the args
        optimizer, scheduler = argparser_opt_scheduler(model, self.args)

        # set the dataloader
        self.trainer.set_with_dataloader(
            train_dataloader=train_dataloader,
            test_dataloader_dict=test_dataloader_dict,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.args.device,
            amp=self.args.amp,
            frequency_save=self.args.frequency_save,
            save_folder_path=self.args.save_path,
            save_prefix='freq_test_wxl',

            # default: False, these setting are for prefetching
            prefetch=args.prefetch,
            prefetch_transform_attr_name="ori_image_transform_in_loading",
            non_blocking=args.non_blocking
        )

        # test the model on clean
        clean_test_loss_avg_over_batch, \
            bd_test_loss_avg_over_batch, \
            test_acc, \
            test_asr, \
            test_ra = self.trainer.test_current_model(
                test_dataloader_dict, args.device
            )

        agg({
            "clean_test_loss_avg_over_batch": clean_test_loss_avg_over_batch,
            "bd_test_loss_avg_over_batch": bd_test_loss_avg_over_batch,
            "test_acc": test_acc,
            "test_asr": test_asr,
            "test_ra": test_ra
        })

        result = {}
        result['model'] = model
        save_defense_result(
            model_name=args.model,
            num_classes=args.num_classes,
            model=model.cpu().state_dict(),
            save_path=args.save_path,
        )
        return result

    def defense(self, result_file):
        self.set_result(result_file)
        self.set_logger()
        result = self.mitigation()
        return result


if __name__ == '__main__':
    # must contain two arguments: yaml_path and result_file
    # i.e. python freq_test_wxl.py --yaml_path ../config/attack/prototype/20-imagenet.yaml --result_file badnet_0_1
    parser = argparse.ArgumentParser(description=sys.argv[0])
    spectral.add_arguments(parser)
    args = parser.parse_args()
    if "result_file" not in args.__dict__:
        args.result_file = 'badnet_0_1'
    elif args.result_file is None:
        args.result_file = 'badnet_0_1'
    if "yaml_path" not in args.__dict__:
        args.yaml_path = './config/defense/freq_test_wxl/20-imagenet.yaml'
    elif args.yaml_path is None:
        args.yaml_path = './config/defense/freq_test_wxl/20-imagenet.yaml'
    spectral_method = spectral(args)

    result = spectral_method.defense(args.result_file)
