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
    @file: freq2_test_wxl.py
    @brief: This file is modified from spectral.py, in order to test frequency domain defense.
    @date: 2021-06-12
    @version: 1.0
    @Author: Xuelin Wei
    @Contact: xuelinwei@seu.edu.cn
'''

'''
This file is modified from spectral.py, in order to test frequency domain defense.
'''

from PIL import Image
from torch import Tensor, torch
from torch.utils.data import Dataset
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
import copy
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


# visual module


def plot_fft(_image):
    to_tensor = transforms.ToTensor()
    _image_tensor = to_tensor(_image)
    if _image_tensor.shape[1] >= 244 or _image_tensor.shape[2] >= 244:
        _image_tensor = transforms.transforms.Resize((224, 224))(_image_tensor)
    _image_fft = fft2(_image_tensor)
    _image_fft = fftshift(_image_fft)
    to_PIL = transforms.ToPILImage()
    _image = to_PIL(_image_tensor)
    _image_fft = _image_fft.sum(axis=0)
    _image_fft = np.abs(_image_fft)
    return _image, _image_fft, _image_tensor


class freq2_test_wxl(defense):

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
        self.clean_model = False

        if 'result_file' in args.__dict__:
            if args.result_file is not None:
                self.set_result(args.result_file)

    def add_arguments(parser):
        parser.add_argument('--device', default='cuda:1',
                            type=str, help='cuda, cpu')

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
        parser.add_argument('--alpha', type=float, default=0.09,
                            help='the percentage of the low frequency part of the image')
        parser.add_argument('--beta', type=float, default=1,
                            help='mask amplification factor')
        parser.add_argument('--replace_mode', type=int, default=0,
                            help='replace mode: 0.amplitude 1.phase 2.both')
        # parser.add_argument('--yaml_path', type=str, default="./config/defense/spectral/config.yaml", help='the path of yaml')

        # # set the parameter for the spectral defense
        # parser.add_argument('--percentile', type=float)
        # parser.add_argument('--target_label', type=int)

    def set_result(self, result_file):
        attack_file = './record/' + result_file
        save_path = './record/' + result_file + '/defense/freq2_test_wxl/'
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
        if os.path.exists(attack_file + '/attack_result.pt'):
            # these code load backdoored model
            self.result = load_attack_result(attack_file + '/attack_result.pt')
            logging.info('backdoored model loaded successfully')
        elif os.path.exists(attack_file + '/info.pickle'):
            # these code load clean model
            # load info
            self.result = {}
            for key, value in torch.load(attack_file + '/info.pickle').items():
                if 'model' in key:
                    self.args.model = value
                elif 'num_classes' in key:
                    self.args.num_classes = value
                elif 'client_optimizer' in key:
                    self.args.client_optimizer = value
                elif 'lr_scheduler' in key:
                    self.args.lr_scheduler = value
                else:
                    print(key, value)
            self.clean_model = True
            logging.info('clean model loaded successfully')
        else:
            assert False, "no attack_result file or info.pickle"

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

        # Very Important !!!
        # the number of workers should be set to 0, because we load a model for trans
        self.args.num_workers = 0

        # setting devices
        if "," in self.device:
            model = torch.nn.DataParallel(
                model,
                # eg. "cuda:2,3,7" -> [2,3,7]
                device_ids=[int(i) for i in self.args.device[5:].split(",")]
            )
            self.args.device = f'cuda:{model.device_ids[0]}'
        logging.info(f'Using device: {self.args.device}')

        if self.clean_model:
            model_file = './record/' + self.args.result_file
            model = generate_cls_model(self.args.model, self.args.num_classes)
            model.load_state_dict(torch.load(model_file + '/clean_model.pth'))
            model.to(self.args.device)

            from utils.aggregate_block.dataset_and_transform_generate import dataset_and_transform_generate
            from utils.bd_dataset_v2 import dataset_wrapper_with_transform

            criterion = argparser_criterion(self.args)
            optimizer, scheduler = argparser_opt_scheduler(model, self.args)
            train_dataset_without_transform, \
                train_img_transform, \
                train_label_transform, \
                test_dataset_without_transform, \
                test_img_transform, \
                test_label_transform = dataset_and_transform_generate(args)

            clean_train_dataset_with_transform = dataset_wrapper_with_transform(
                train_dataset_without_transform,
                train_img_transform,
                train_label_transform,
            )

            clean_test_dataset_with_transform = dataset_wrapper_with_transform(
                test_dataset_without_transform,
                test_img_transform,
                test_label_transform,
            )

            data_clean_train_loader = torch.utils.data.DataLoader(clean_train_dataset_with_transform, batch_size=self.args.batch_size,
                                                                  num_workers=self.args.num_workers, drop_last=False, shuffle=False, pin_memory=args.pin_memory)

            data_clean_test_loader = torch.utils.data.DataLoader(clean_test_dataset_with_transform, batch_size=self.args.batch_size,
                                                                 num_workers=self.args.num_workers, drop_last=False, shuffle=False, pin_memory=args.pin_memory)

            # random choose a image from test dataset
            length = len(clean_test_dataset_with_transform)
            index = np.random.randint(0, length)
            low_freq_image = test_dataset_without_transform[index][0]
            low_freq_image = transforms.Resize(
                (self.args.input_height, self.args.input_width))(low_freq_image)
            low_freq_image = transforms.ToTensor()(low_freq_image)
            # prepare the low frequency substitution module
            low_freq_sub = low_freq_substitution(
            self.args.input_height, self.args.input_width, low_freq_image, self.args.alpha, self.args.beta, self.args.replace_mode)
            # add the low frequency substitution module to the transform
            logging.info("changing the transform of the bd dataset")
            test_trans = copy.deepcopy(test_img_transform)
            temp = test_trans.transforms[-1]
            test_trans.transforms[-1] = low_freq_sub
            test_trans.transforms.append(temp)
            logging.info("new transforms", test_trans)

            transformed_test_dataset_with_transform = dataset_wrapper_with_transform(
                test_dataset_without_transform,
                test_trans,
                test_label_transform,
            )

            data_transformed_test_loader = torch.utils.data.DataLoader(transformed_test_dataset_with_transform, batch_size=self.args.batch_size,
                                                                       num_workers=self.args.num_workers, drop_last=False, shuffle=False, pin_memory=args.pin_memory)

            # # TODO:this part is for visualization
            # # visualize the transformed image
            # import matplotlib.pyplot as plt
            # from tqdm import tqdm
            # transformed_test_savepath = self.args.save_path+'/transformed_test'
            # if os.path.exists(transformed_test_savepath) == False:
            #     os.makedirs(transformed_test_savepath)
            # for index, (img, label) in tqdm(enumerate(transformed_test_dataset_with_transform)):
            #     img.to(self.args.device)
            #     to_PIL = transforms.ToPILImage()
            #     denormalizer = transforms.Normalize(
            #         mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
            #     img = denormalizer(img)
            #     img = to_PIL(img)
            #     img.save(f'{transformed_test_savepath}/{index}.png')

            # in order to logging the result together
            agg = Metric_Aggregator()

            # put the model into trainer handle
            self.set_trainer(model)

            # set the dataloader dict
            test_dataloader_dict = {
                "Clean": data_clean_test_loader,
                "Transformed": data_transformed_test_loader,
            }
            # test_dataloader_dict['bd_test_dataloader'] = data_bd_test_loader

            self.trainer.set_with_dataloader(
                train_dataloader=data_clean_train_loader,
                test_dataloader_dict=test_dataloader_dict,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device=self.args.device,
                amp=self.args.amp,
                frequency_save=self.args.frequency_save,
                save_folder_path=self.args.save_path,
                save_prefix='freq2_test_wxl',

                # default: False, these setting are for prefetching
                prefetch=args.prefetch,
                prefetch_transform_attr_name="ori_image_transform_in_loading",
                non_blocking=args.non_blocking
            )

            # test the model on clean
            metrics = self.trainer.test_all_inner_dataloader()
            clean_metrics = metrics["Clean"]
            transformed_metrics = metrics["Transformed"]
            print("##################clean_metrics##################")
            print(clean_metrics)
            print("##################transformed_metrics##################")
            print(transformed_metrics)
            assert False

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
            assert False, "clean model is not supported yet"

        # prepare the model and device
        model = generate_cls_model(self.args.model, self.args.num_classes)
        model.load_state_dict(self.result['model']) 

        model.to(self.args.device)
        # Setting up the data and the model
        train_trans = get_transform(
            self.args.dataset, *([self.args.input_height, self.args.input_width]), train=True)
        test_img_transform = get_transform(
            self.args.dataset, *([self.args.input_height, self.args.input_width]), train=False)

        # set the clean trainning dataset and its transform
        data_clean_trainset = self.result['clean_train']
        data_clean_trainset.wrap_img_transform = train_trans
        data_clean_trainset_loader = torch.utils.data.DataLoader(data_clean_trainset, batch_size=self.args.batch_size,
                                                                 num_workers=self.args.num_workers, drop_last=False, shuffle=True, pin_memory=args.pin_memory)

        # set a bd test dataset and its transform to compare
        data_bd_testset_without_transform = copy.deepcopy(
            self.result['bd_test'])
        data_bd_testset_without_transform.wrap_img_transform = test_img_transform

        # set a clean test dataset and its transform to compare
        data_clean_testset_without_transform = copy.deepcopy(
            self.result['clean_test'])
        data_clean_testset_without_transform.wrap_img_transform = test_img_transform

        # define a new transform for the frequency domain processing
        # original transform: 0.resize 1.totensor 2.normalize
        # new transform: 0.resize 1.totensor 2.normalize(0.5) 3.reconstruction 4.denomalize 5.resize 6.normalize
        # in oder to get a pil image, set wrap_img_transform to None
        from FA_VAE.favae_scripts.load_favae import loadfavae
        print('loading model ............')
        reconstruction = loadfavae()
        test_trans = [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            reconstruction,
            transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),
            transforms.ToPILImage(),
        ]
        test_trans.extend(copy.deepcopy(test_img_transform.transforms))   
        test_trans = transforms.Compose(test_trans)
        
        logging.info("changing the transform of both datasets")
        logging.info("new transforms", test_trans)

        # set bd dataset, transform and dataloader
        data_bd_testset = self.result['bd_test']
        data_bd_testset.wrap_img_transform = test_trans
        data_bd_test_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=self.args.batch_size,
                                                          num_workers=self.args.num_workers, drop_last=False, shuffle=False, pin_memory=args.pin_memory)

        # set clean dataset, transform and dataloader
        data_clean_testset = self.result['clean_test']
        data_clean_testset.wrap_img_transform = test_trans
        data_clean_test_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=self.args.batch_size,
                                                             num_workers=self.args.num_workers, drop_last=False, shuffle=False, pin_memory=args.pin_memory)

        '''
            This part is for visualization
        '''
        # in order to compare the difference between the original image and the transformed image
        import matplotlib.pyplot as plt
        from tqdm import tqdm
        from focal_frequency_loss import FocalFrequencyLoss as FFL
        import seaborn as sns
        import datetime
        experiment_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        transformed_test_savepath = self.args.save_path+f'/results/{experiment_time}'
        if os.path.exists(transformed_test_savepath) == False:
            os.makedirs(transformed_test_savepath)
        len_bd = len(data_bd_testset_without_transform)
        len_clean = len(data_clean_testset_without_transform)
        distance = len_clean-len_bd
        # random select 5 images from the test dataset
        random_index = np.random.randint(0, len_bd, 1)
        for _, dataset_index in tqdm(enumerate(random_index)):
            to_PIL = transforms.ToPILImage()
            denormalizer = transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])

            # there are some bugs need to be fixed:
            #   - for the cifar10, the clean dataset and bd dataset is not matched
            #   - it's difficult to get coresponding clean sample for a bd sample

            img = data_clean_testset_without_transform[distance +
                                                       dataset_index][0]
            transformed_img = data_clean_testset[distance+dataset_index][0]
            bd_img = data_bd_testset_without_transform[dataset_index][0]
            transformed_bd_img = data_bd_testset[dataset_index][0]

            img.to(self.args.device)
            bd_img.to(self.args.device)
            transformed_img.to(self.args.device)
            transformed_bd_img.to(self.args.device)

            imgs = []
            imgs.append(to_PIL(denormalizer(img)))
            imgs.append(to_PIL(denormalizer(transformed_img)))
            imgs.append(to_PIL(denormalizer(bd_img)))
            imgs.append(to_PIL(denormalizer(transformed_bd_img)))

            length = len(imgs)
            fig = plt.figure(figsize=(50, length*10))
            ffl = FFL(loss_weight=1.0, alpha=1.0)
            original_image_fft = 0
            orignal_image_tensor = 0
            column = 10
            for index, image in enumerate(imgs):
                image, image_fft, image_tensor = plot_fft(image)
                if index == 0:
                    original_image_fft = image_fft
                    orignal_image_tensor = image_tensor.clone()
                    orignal_image_tensor = orignal_image_tensor.unsqueeze(0)
                fig.add_subplot(length, column, index*column+1)
                plt.imshow(image)
                plt.title(f'{dataset_index}', fontsize=40)
                plt.axis('off')
                # show original image on the above row
                fig.add_subplot(length, column, index*column+2)
                sns.heatmap(np.log(original_image_fft+1),
                            cmap='viridis', cbar=False)
                plt.axis('off')
                plt.title('log view of orgin fft', fontsize=20)
                # # show fft heatmap on the below row
                fig.add_subplot(length, column, index*column+3)
                sns.heatmap(np.log(image_fft+1), cmap='viridis', cbar=False)
                plt.axis('off')
                plt.title('log view of backdoored fft', fontsize=20)
                fig.add_subplot(length, column, index*column+4)
                fft_diff = image_fft - original_image_fft
                sns.heatmap(np.abs(fft_diff), cmap='viridis', cbar=False)
                plt.axis('off')
                image_tensor = image_tensor.unsqueeze(0)
                fflloss = ffl(image_tensor, orignal_image_tensor)
                plt.title(f'FFL: {fflloss:.8f}', fontsize=40)
                fig.add_subplot(length, column, index*column+5)
                fft_diff = image_fft - original_image_fft
                sns.heatmap(np.log(np.abs(fft_diff)+1),
                            cmap='viridis', cbar=False)
                plt.axis('off')
                plt.title('diff log view', fontsize=20)
                fig.add_subplot(length, column, index*column+6)
                fft_diff_1 = fft_diff.detach().clone()
                fft_diff_1[fft_diff < 0] = 0
                sns.heatmap(np.log(np.abs(fft_diff_1)+1),
                            cmap='viridis', cbar=False)
                plt.axis('off')
                plt.title('Positive diff', fontsize=20)
                fig.add_subplot(length, column, index*column+7)
                fft_diff_2 = fft_diff.detach().clone()
                fft_diff_2[fft_diff > 0] = 0
                sns.heatmap(np.log(np.abs(fft_diff_2)+1),
                            cmap='viridis', cbar=False)
                plt.axis('off')
                plt.title('Negative diff', fontsize=20)
                fig.add_subplot(length, column, index*column+8)
                ffta = fft_diff.detach().clone().numpy()
                ffta = np.square(ffta)
                # get center of the fft
                h, w = ffta.shape
                center = (h-1)/2, (w-1)/2
                # get the radius of the fft
                max_radius = center[0]**2 + center[1]**2
                max_radius = int(np.ceil(np.sqrt(max_radius)))
                # create a blank array to store the fft's distribution
                fft_distribution = np.zeros(max_radius, dtype=np.float32)
                # calculate the fft's distribution
                for i in range(h):
                    for j in range(w):
                        radius = int(
                            np.ceil(np.sqrt((i-center[0])**2 + (j-center[1])**2)))
                        fft_distribution[radius-1] += ffta[i, j]
                # plot the fft's distribution
                plt.plot(fft_distribution)
                fig.add_subplot(length, column, index*column+9)
                plt.plot(fft_distribution[:int(
                    np.floor(min(center[0], center[1])))])
                fig.add_subplot(length, column, index*column+10)
                plt.plot(fft_distribution[10:int(
                    np.floor(min(center[0], center[1])))])
            fig.savefig(f'{transformed_test_savepath}/{dataset_index}.png')

        # set the dataloader dict
        test_dataloader_dict = {}
        test_dataloader_dict['clean_test_dataloader'] = data_clean_test_loader
        test_dataloader_dict['bd_test_dataloader'] = data_bd_test_loader

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
            train_dataloader=data_clean_trainset_loader,
            test_dataloader_dict=test_dataloader_dict,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.args.device,
            amp=self.args.amp,
            frequency_save=self.args.frequency_save,
            save_folder_path=self.args.save_path,
            save_prefix='freq2_test_wxl',

            # default: False, these setting are for prefetching
            prefetch=args.prefetch,
            prefetch_transform_attr_name="ori_image_transform_in_loading",
            non_blocking=args.non_blocking
        )

        # test the model
        # important: the bd test dataset used to test may contain clean data whose label is the target label
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

        # save to csv
        import pandas as pd
        # first, read the result.csv before
        csv_result_save_path = self.args.save_path+'/results/result.csv'
        result_dict = {
            "experiment_time": experiment_time,
            "replace_mode": self.args.replace_mode,
            "alpha": self.args.alpha,
            "beta": self.args.beta,
            "test_acc": test_acc,
            "test_asr": test_asr,
            "test_ra": test_ra,
        }
        if os.path.exists(csv_result_save_path):
            df = pd.read_csv(csv_result_save_path)
        else:
            df = pd.DataFrame()

        df = df.append(result_dict, ignore_index=True)
        df.to_csv(csv_result_save_path, index=False)
        return result

    def defense(self, result_file):
        self.set_result(result_file)
        self.set_logger()
        result = self.mitigation()
        return result


if __name__ == '__main__':
    # must contain two arguments: yaml_path and result_file
    # i.e. python freq2_test_wxl.py --yaml_path ../config/attack/prototype/20-imagenet.yaml --result_file badnet_0_1
    parser = argparse.ArgumentParser(description=sys.argv[0])
    freq2_test_wxl.add_arguments(parser)
    args = parser.parse_args()
    # if "result_file" not in args.__dict__:
    #     args.result_file = '20240119_204623_prototype_attack_prototype_A28B'
    # elif args.result_file is None:
    #     args.result_file = '20240119_204623_prototype_attack_prototype_A28B'
    if "result_file" not in args.__dict__:
        args.result_file = 'ssba_0_1'
    elif args.result_file is None:
        args.result_file = 'ssba_0_1'
    if "yaml_path" not in args.__dict__:
        args.yaml_path = './config/defense/freq_test_wxl/20-imagenet.yaml'
    elif args.yaml_path is None:
        args.yaml_path = './config/defense/freq_test_wxl/20-imagenet.yaml'
    spectral_method = freq2_test_wxl(args)

    result = spectral_method.defense(args.result_file)
