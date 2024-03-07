#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
import time
import torch
import numpy as np
from tqdm import tqdm
import os
import argparse
import sys
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
from base import defense
sys.path.append(os.getcwd())
from pprint import  pformat
import logging
import time
from utils.save_load_attack import load_attack_result, save_defense_result
from utils.log_assist import get_git_info
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from torchvision.utils import save_image
import cv2
from torchvision import transforms


class Filter(defense):  
    def __init__(self, args):
        super(Filter, self).__init__()
        self.args = args
    
    def set_args(self):
        pass

    
    def add_arguments(parser):
        parser.add_argument('--device', type=str, help='cuda, cpu', default="cuda:3")
        parser.add_argument('--result_file', type=str, help='the location of result', default='wanet_gaussblur')  
        parser.add_argument('--log', type=str, help='the location of log')
        parser.add_argument('--random_seed', type=int, help='random seed', default=0)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument("--num_workers", type=int, default=2)
        parser.add_argument('--target_label', type=int, default=0)
        parser.add_argument('--model', type=str, default='preactresnet18')
        parser.add_argument('--num_classes', type=int, default=43)
        parser.add_argument('--cutoff_percentage', type=int, default=90)
        parser.add_argument('--filtered_type', type=str, default='specific')
        parser.add_argument('--low_freq', type=float, default=0)
        parser.add_argument('--high_freq', type=float, default=0.01)

        return parser 
    
    def set_result(self, result_file):
        attack_file = 'record/' + self.args.result_file
        self.attack_file = attack_file
        save_path = 'record/' + result_file + '/defense/filter/'
        if not (os.path.exists(save_path)):
            os.makedirs(save_path)
        # assert(os.path.exists(save_path))    
        # self.args.save_path = save_path
        if self.args.log is None:
            self.args.log = save_path + 'log/'
            if not (os.path.exists(self.args.log)):
                os.makedirs(self.args.log)  
        self.result = load_attack_result(attack_file + '/attack_result.pt')  
        
    def set_logger(self):
        args = self.args
        logFormatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
        )
        logger = logging.getLogger()

        fileHandler = logging.FileHandler(args.log + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
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
        # self.device = torch.device(
        #     (
        #         f"cuda:{[int(i) for i in self.args.device[5:].split(',')][0]}" if "," in self.args.device else self.args.device
        #         # since DataParallel only allow .to("cuda")
        #     ) if torch.cuda.is_available() else "cpu"
        # )
        self.device = self.args.device
    
    '''def zigzag_scan(self, spectrum):
        B, M, N = spectrum.shape
        zigzag_result = np.zeros((B, M * N))
        
        index = 0
        row = 0
        col = 0
        
        while index < M * N:
            zigzag_result[:, index] = spectrum[:, row, col]
            
            if (row + col) % 2 == 0:  # 向上移动
                if col == N-1:
                    row += 1
                elif row == 0:
                    col += 1
                else:
                    row -= 1
                    col += 1
            else:  # 向下移动
                if row == M-1:
                    col += 1
                elif col == 0:
                    row += 1
                else:
                    row += 1
                    col -= 1
            
            index += 1
        
        return zigzag_result'''
    
    '''def zigzag_scan(SELF, arr):
        row, col = arr.shape[1:]
        result = np.zeros((arr.shape[0], row * col))
        for i in range(arr.shape[0]):
            result[i] = np.concatenate([np.diagonal(arr[i], offset=j)[::(2*(j % 2)-1)] for j in range(1 - row, col)])
        return result'''
        
    def zigzag_scan(self, spectrum):
        B, M, N = spectrum.shape
        zigzag_result = np.zeros((B, M * N))           
        for i in range(B):
            index = 0
            row = 0
            col = 0
            while index < M * N:
                zigzag_result[i, index] = spectrum[i, row, col]              
                if (row + col) % 2 == 0:  # 向上移动
                    if col == N-1:
                        row += 1
                    elif row == 0:
                        col += 1
                    else:
                        row -= 1
                        col += 1
                else:  # 向下移动
                    if row == M-1:
                        col += 1
                    elif col == 0:
                        row += 1
                    else:
                        row += 1
                        col -= 1           
                index += 1       
        return zigzag_result

    
    def zigzag_inverse(self, arr, original_shape):
        result = np.zeros(original_shape)
        height, width = original_shape[1:]
        for i in range(arr.shape[0]):
            zigzag = arr[i]
            row, col = 0, 0
            direction = 1 # 初始方向为向下
            for k in range(height * width):
                result[i, row, col] = zigzag[k]
                if direction == 1:  # 向下
                    if col == width - 1:
                        row += 1
                        direction = -1
                    elif row == 0:
                        col += 1
                        direction = -1
                    else:
                        row -= 1
                        col += 1
                else:  # 向上
                    if row == height - 1:
                        col += 1
                        direction = 1
                    elif col == 0:
                        row += 1
                        direction = 1
                    else:
                        row += 1
                        col -= 1
        return result

        
    def bandpass_filter(self, images, low_freq, high_freq):
        # 对图像进行DCT变换
        dct_image = np.array([cv2.dct(image) for image in images])
        # dct_image = cv2.dct(images.astype(np.float32), flags=cv2.DCT_ROWS)
        
        # 获取DCT频谱图的尺寸
        _, M, N = dct_image.shape
        
        # 对频谱图进行 Zigzag 扫描
        zigzag_result = self.zigzag_scan(dct_image)
        
        # 计算频率范围对应的索引范围
        fs = 1  # 采样率
        len_ = zigzag_result.shape[1]  # 频谱图的长度
        low_index = round(low_freq * len_ / fs)
        high_index = round(high_freq * len_ / fs)
        
        # 将截断频率范围之外的频率置零
        filtered_zigzag = zigzag_result.copy()
        filtered_zigzag[:, low_index:high_index] = 0
        
        dct_image_filtered = self.zigzag_inverse(filtered_zigzag, dct_image.shape)

        # 对滤波后的DCT系数进行逆变换
        idct_image = np.array([cv2.idct(dct) for dct in dct_image_filtered])
        
        idct_image = np.clip(idct_image, 0, 1)
        
        return idct_image, dct_image, dct_image_filtered
    
    def apply_RGB_filter(self, rgb_image, low_freq, high_freq):
        rgb_image = rgb_image.cpu().numpy()
        filtered_red_channel, red_dct_image, red_amplitude_spectrum  = self.bandpass_filter(rgb_image[:,0], low_freq, high_freq)
        filtered_green_channel, green_dct_image, green_amplitude_spectrum = self.bandpass_filter(rgb_image[:,1], low_freq, high_freq)
        filtered_blue_channel, blue_dct_image, blue_amplitude_spectrum = self.bandpass_filter(rgb_image[:,2], low_freq, high_freq)
        rgb_amplitude_spectrum = np.stack((red_dct_image, green_dct_image, blue_dct_image), axis=1)
        filtered_rgb_amplitude_spectrum = np.stack((red_amplitude_spectrum, green_amplitude_spectrum, blue_amplitude_spectrum), axis=1)
        filtered_rgb_image = np.stack((filtered_red_channel, filtered_green_channel, filtered_blue_channel), axis=1)
        return filtered_rgb_image, rgb_amplitude_spectrum, filtered_rgb_amplitude_spectrum

    
    def mitigate(self):
        self.set_devices()
        fix_random(self.args.random_seed)
        args = self.args
        result = self.result
        
        model = generate_cls_model(self.args.model,self.args.num_classes)
        model.load_state_dict(self.result['model'])
        model.to(self.device)
        self.model = model
        
        clean_test_with_trans = result["clean_test"]
        clean_test_with_trans.original_index_array = np.arange(len(clean_test_with_trans.wrapped_dataset))
        clean_test_with_trans.original_index_array =  clean_test_with_trans.original_index_array[np.where(np.array(clean_test_with_trans.wrapped_dataset.labels) != args.target_label)[0]]
        
        bd_test_with_trans = result["bd_test"]
        args.batch_size=128
        clean_data_loader = torch.utils.data.DataLoader(clean_test_with_trans, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
        bd_data_loader = torch.utils.data.DataLoader(bd_test_with_trans, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
        
        
        total = 0
        ori_ACC = 0
        ori_ASR = 0
        ori_bd_robust = 0
        ASR = 0
        ACC = 0
        bd_robust = 0
        to_gray = transforms.Grayscale()    
        pbar = tqdm(total=round(len(clean_test_with_trans)/args.batch_size), desc='Filter')
        ##################################################

        for i, data in enumerate(zip(clean_data_loader,bd_data_loader)):
            model.eval()
            cln_imgs, labels = data[0]
            cln_imgs = cln_imgs
            labels = labels.to(self.device)
            bd_imgs = data[1][0]
            target_labels = (torch.ones_like(labels)*args.target_label).to(self.device)
                        
            # gray_cln_imgs = to_gray(cln_imgs)    
            # gray_bd_imgs = to_gray(bd_imgs)
            
            filtered_cln_image, _, _ = self.apply_RGB_filter(cln_imgs, args.low_freq, args.high_freq)
            filtered_bd_image, rgb_amplitude_spectrum, filter_rgb_amplitude_spectrum = self.apply_RGB_filter(bd_imgs, args.low_freq, args.high_freq)
            filtered_cln_image = torch.from_numpy(filtered_cln_image).float().to(self.device) # filtered_cln_image.to(self.device)
            filtered_bd_image = torch.from_numpy(filtered_bd_image).float().to(self.device) # filtered_bd_image.to(self.device)
            
            # if args.filtered_type == 'gaussian':                            
            #     filtered_cln_image = torch.from_numpy(np.stack((cv2.blur(cln_imgs[:,0].numpy(), (3, 3)), cv2.blur(cln_imgs[:,1].numpy(), (3, 3)), cv2.blur(cln_imgs[:,2].numpy(), (3, 3))), axis=1)).float().to(self.device)
            #     filtered_bd_image = torch.from_numpy(np.stack((cv2.blur(bd_imgs[:,0].numpy(), (3, 3)), cv2.blur(bd_imgs[:,1].numpy(), (3, 3)), cv2.blur(bd_imgs[:,2].numpy(), (3, 3))), axis=1)).float().to(self.device)
                          
            outputs_cln = model(filtered_cln_image)
            _, predicted_ori_filter = torch.max(outputs_cln, 1) # predicted_ori_filter is the tensor stored original predicted before GAN
            ACC += (predicted_ori_filter == labels).sum().item()

            outputs_bd = model(filtered_bd_image)
            _, predicted_bd_filter = torch.max(outputs_bd, 1)
            ASR += (predicted_bd_filter == target_labels).sum().item()
            bd_robust += (predicted_bd_filter == labels).sum().item()
            
            ################################################################
            cln_imgs = cln_imgs.to(self.device)
            bd_imgs = bd_imgs.to(self.device)
            
            outputs_cln = model(cln_imgs)
            _, predicted_ori = torch.max(outputs_cln, 1) # predicted_ori is the tensor stored original predicted before GAN
            ori_ACC += (predicted_ori == labels).sum().item()

            outputs_bd = model(bd_imgs)
            _, predicted_bd = torch.max(outputs_bd, 1)
            ori_ASR += (predicted_bd == target_labels).sum().item()
            ori_bd_robust += (predicted_bd == labels).sum().item()
            
            total += labels.shape[0]
            pbar.update()
            
            # title_fontdict = {'fontsize': 6}
            # import matplotlib.pyplot as plt
            # plt.subplot(2, 3, 1)
            # plt.axis("off")
            # plt.imshow(cv2.cvtColor(np.transpose(cln_imgs[0].cpu().numpy(), (1, 2, 0)), cv2.COLOR_BGR2RGB))
            # plt.title(f'Original Image  Prediction:{predicted_ori[0].item()}', fontdict=title_fontdict)
            
            # plt.subplot(2, 3, 2)
            # plt.axis("off")
            # plt.imshow(cv2.cvtColor(np.transpose(bd_imgs[0].cpu().numpy(), (1, 2, 0)), cv2.COLOR_BGR2RGB))
            # plt.title(f'Backdoor Image  Prediction:{predicted_bd[0].item()}', fontdict=title_fontdict)
            
            # plt.subplot(2, 3, 3)
            # plt.axis("off")
            # plt.imshow(cv2.cvtColor(np.transpose(1*(cln_imgs[0]-bd_imgs[0]).cpu().numpy(), (1, 2, 0)), cv2.COLOR_BGR2RGB))
            # plt.title('Trigger', fontdict=title_fontdict)
            
            # plt.subplot(2, 3, 4)
            # plt.axis("off")
            # plt.imshow(np.transpose(rgb_amplitude_spectrum[0], (1, 2, 0)), cmap='gray')
            # plt.title('Original Amplitude Spectrum', fontdict=title_fontdict)
            
            # plt.subplot(2, 3, 5)
            # plt.axis("off")
            # plt.imshow(np.transpose(filter_rgb_amplitude_spectrum[0], (1, 2, 0)), cmap='gray')
            # plt.title('Filtered Amplitude Spectrum', fontdict=title_fontdict)
            
            # plt.subplot(2, 3, 6)
            # plt.axis("off")
            # plt.imshow(cv2.cvtColor(np.transpose(filtered_bd_image[0].cpu().numpy(), (1, 2, 0)), cv2.COLOR_BGR2RGB))
            # plt.title(f'{args.cutoff_percentage}% Filtered Image Prediction:{predicted_bd_filter[0].item()}', fontdict=title_fontdict)
            # plt.subplot(2, 3, 6)
            # plt.axis("off")
            # plt.imshow(cv2.cvtColor(np.transpose((bd_imgs[0].cpu()-filtered_bd_image[0].cpu()).numpy(), (1, 2, 0)), cv2.COLOR_BGR2RGB))
            # plt.title(f'bd minus filtered bd', fontdict=title_fontdict)
            # plt.savefig('dct.png')
        pbar.close()
        
        with open(f'results_{args.filtered_type}_{args.cutoff_percentage}.txt', 'a+') as file:         
            print('##################################################')
            print('# Before Filter:')
            print('Accuracy of benigh inputs before Filter: %.3f %%' % (100 * ori_ACC / total))
            print('Attack success rate before Filter: %.3f %%' % (100 * ori_ASR / total))
            print('Backdoor Robustness after Filter: %.3f %%' % (100 * ori_bd_robust / total))
            print('\n# After Filter:\n')
            print('Accuracy of sanitized input after Filter: %.3f %%' % (100 * ACC / total))
            print('Atack Success rate after Filter: %.3f %%' % (100 * ASR / total))
            print('Backdoor Robustness after Filter: %.3f %%' % (100 * bd_robust / total))
            # file.write('\n# After Filter:\n')
            # file.write('Accuracy of sanitized input after Filter: %.3f %%\n' % (100 * ACC / total))
            # file.write('Atack Success rate after Filter: %.3f %%\n' % (100 * ASR / total))
            # file.write('Backdoor Robustness after Filter: %.3f %%\n' % (100 * bd_robust / total))
        file.close()
        
        
    def defense(self,result_file):
        self.set_result(result_file)
        self.set_logger()
        result = self.mitigate()
        return result
              
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=sys.argv[0])
    Filter.add_arguments(parser)
    args = parser.parse_args()
    filter = Filter(args)
    if "result_file" not in args.__dict__:
        args.result_file = 'defense_test_badnet'
    elif args.result_file is None:
        args.result_file = 'defense_test_badnet'
    
    with open(f'results_{args.filtered_type}_{args.cutoff_percentage}.txt', 'a+') as file:
        file.write('##################################################\n')
        file.write("==================Load result from: "+args.result_file+"========================\n")
        file.write(str(args))
    file.close()
            
    filter.defense(args.result_file)