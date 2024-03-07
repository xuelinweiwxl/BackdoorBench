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
import csv
import pandas as pd
from datetime import datetime
from torchvision import transforms


class Filter(defense):  
    def __init__(self, args):
        super(Filter, self).__init__()
        self.args = args
    
    def set_args(self):
        pass

    
    def add_arguments(parser):
        parser.add_argument('--device', type=str, help='cuda, cpu', default="cuda")
        parser.add_argument('--result_file', type=str, help='the location of result', default='wanet_0_1')  
        parser.add_argument('--log', type=str, help='the location of log')
        parser.add_argument('--random_seed', type=int, help='random seed', default=0)
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument("--num_workers", type=int, default=2)
        parser.add_argument('--target_label', type=int, default=0)
        parser.add_argument('--model', type=str, default='preactresnet18')
        parser.add_argument('--num_classes', type=int, default=20)
        parser.add_argument('--cutoff_percentage', type=int, default=90)
        parser.add_argument('--filtered_type', type=str, default='high_ideal')

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
        
    def apply_filter(self, image, filter_type, cutoff_percentage):
        # 转换为灰度图像
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = image
        
        # 将灰度图像转换为频域
        fft = np.fft.fft2(gray_image)
        # 将频谱移到中心
        fft_shift = np.fft.fftshift(fft)
        
        # 计算频谱
        # magnitude_spectrum = 20 * np.log(cv2.magnitude(fft_shift[:, :, 0], fft_shift[:, :, 1]))
        gray_amplitude_spectrum = np.log(1 + np.abs(fft_shift))
        
        # 获取图像尺寸
        _, rows, cols = gray_image.shape
        
        # 计算中心点坐标
        center_row, center_col = int(rows / 2), int(cols / 2)
        
        # 计算截止频率
        import math
        cutoff = int((cutoff_percentage / 100) * min(rows, cols)/2)
        # cutoff = int((cutoff_percentage / 100) * min(math.sqrt(2)*rows, math.sqrt(2)*cols)/2)
        cutoff_high = int(rows/2 - cutoff)
        
        # 根据滤波器类型创建掩膜
        if filter_type == 'high_ideal':
            mask_low = np.ones((rows, cols), np.uint8)
            mask_low[center_row - cutoff:center_row + cutoff, center_col - cutoff:center_col + cutoff] = 0
            mask = mask_low
            # mask_low = np.ones((rows, cols), np.uint8)
            # mask_low[center_row - cutoff:center_row + cutoff, center_col - cutoff:center_col + cutoff] = 0
            
            # mask_high = np.zeros((rows, cols), np.uint8)
            # mask_high[center_row - cutoff_high:center_row + cutoff_high, center_col - cutoff_high:center_col + cutoff_high] = 1
            # mask = mask_high|mask_low
        elif filter_type == 'gaussian':
            x = np.arange(cols) - center_col
            y = np.arange(rows) - center_row
            xx, yy = np.meshgrid(x, y)           
            sigma = cutoff / 2.3548  # 根据截止频率计算高斯分布的标准差
            distance = np.sqrt(xx**2 + yy**2)   # 计算距离中心点的距离           
            gaussian = np.exp(-(distance**2) / (2 * sigma**2))  # 创建高斯滤波器
            mask = gaussian
        elif filter_type == 'specific':  
            mask = np.ones((rows, cols), np.uint8)
            mask[center_row - cutoff-1:center_row + cutoff + 1,center_row - cutoff-1]=0
            mask[center_row - cutoff-1:center_row + cutoff + 1,center_row + cutoff]=0
            mask[center_row - cutoff-1,center_row - cutoff-1:center_row + cutoff + 1]=0
            mask[center_row + cutoff,center_row - cutoff-1:center_row + cutoff + 1]=0
        else:
            raise Exception('Filter type not supported.')

        
        # 应用掩膜
        filtered_shift = fft_shift * mask
        
        # 将频域转换回空域
        filtered_fft = np.fft.ifftshift(filtered_shift)
        filtered_gray_image = np.real(np.fft.ifft2(filtered_fft))
        # filtered_image = np.abs(filtered_image)
        # 将灰度图像转换为RGB图像
        # filtered_image = cv2.cvtColor(np.uint8(filtered_gray_image), cv2.COLOR_GRAY2RGB)
        filtered_image = filtered_gray_image
        
        
        # 将图像范围限制在0到255之间
        filtered_image = np.clip(filtered_image, 0, 1)
    
        return filtered_image, gray_amplitude_spectrum
    
    def apply_RGB_filter(self, rgb_image, filter_type, cutoff_percentage):
        filtered_red_channel, red_amplitude_spectrum  = self.apply_filter(rgb_image[:,0], filter_type, cutoff_percentage)
        filtered_green_channel, green_amplitude_spectrum = self.apply_filter(rgb_image[:,1], filter_type, cutoff_percentage)
        filtered_blue_channel, blue_amplitude_spectrum = self.apply_filter(rgb_image[:,2], filter_type, cutoff_percentage)
        rgb_amplitude_spectrum = np.stack((red_amplitude_spectrum, green_amplitude_spectrum, blue_amplitude_spectrum), axis=1)
        filtered_rgb_image = np.stack((filtered_red_channel, filtered_green_channel, filtered_blue_channel), axis=1)
        return filtered_rgb_image, rgb_amplitude_spectrum

    
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
        #WXL: 20-ImageNet don't have label
        if 'imagenet' in clean_test_with_trans.root:
            class_to_idx = clean_test_with_trans.wrapped_dataset.class_to_idx
            #WXL: find class_name
            class_to_idx = clean_test_with_trans.wrapped_dataset.class_to_idx
            for k,v in class_to_idx.items():
                if v == args.target_label:
                    target_class_name = k
            clean_test_with_trans.original_index_array =  clean_test_with_trans.original_index_array[np.where(np.array(clean_test_with_trans.wrapped_dataset.targets) != args.target_label)[0]]
        else:
            clean_test_with_trans.original_index_array =  clean_test_with_trans.original_index_array[np.where(np.array(clean_test_with_trans.wrapped_dataset.labels) != args.target_label)[0]]
        
        bd_test_with_trans = result["bd_test"]
        # args.batch_size=128
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
        print(len(clean_test_with_trans))
        print(len(bd_test_with_trans)) 
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
            
            filtered_cln_image, rgb_amplitude_spectrum = self.apply_RGB_filter(cln_imgs, args.filtered_type, args.cutoff_percentage)
            filtered_bd_image, _ = self.apply_RGB_filter(bd_imgs, args.filtered_type, args.cutoff_percentage)
            filtered_cln_image = torch.from_numpy(filtered_cln_image).float().to(self.device) # filtered_cln_image.to(self.device)
            filtered_bd_image = torch.from_numpy(filtered_bd_image).float().to(self.device) # filtered_bd_image.to(self.device)
            
            # if args.filtered_type == 'gaussian':                            
            #     filtered_cln_image = torch.from_numpy(np.stack((cv2.blur(cln_imgs[:,0].numpy(), (3, 3)), cv2.blur(cln_imgs[:,1].numpy(), (3, 3)), cv2.blur(cln_imgs[:,2].numpy(), (3, 3))), axis=1)).float().to(self.device)
            #     filtered_bd_image = torch.from_numpy(np.stack((cv2.blur(bd_imgs[:,0].numpy(), (3, 3)), cv2.blur(bd_imgs[:,1].numpy(), (3, 3)), cv2.blur(bd_imgs[:,2].numpy(), (3, 3))), axis=1)).float().to(self.device)
                          
            outputs_cln = model(filtered_cln_image)
            _, predicted_ori = torch.max(outputs_cln, 1) # predicted_ori is the tensor stored original predicted before GAN
            ACC += (predicted_ori == labels).sum().item()

            outputs_bd = model(filtered_bd_image)
            _, predicted = torch.max(outputs_bd, 1)
            ASR += (predicted == target_labels).sum().item()
            bd_robust += (predicted == labels).sum().item()
            
            ################################################################
            cln_imgs = cln_imgs.to(self.device)
            bd_imgs = bd_imgs.to(self.device)
            
            outputs_cln = model(cln_imgs)
            _, predicted_ori = torch.max(outputs_cln, 1) # predicted_ori is the tensor stored original predicted before GAN
            ori_ACC += (predicted_ori == labels).sum().item()

            outputs_bd = model(bd_imgs)
            _, predicted = torch.max(outputs_bd, 1)
            ori_ASR += (predicted == target_labels).sum().item()
            ori_bd_robust += (predicted == labels).sum().item()
            
            total += labels.shape[0]
            pbar.update()
            
            title_fontdict = {'fontsize': 6}
            import matplotlib.pyplot as plt
            plt.subplot(1, 4, 1)
            plt.axis("off")
            plt.imshow(cv2.cvtColor(np.transpose(bd_imgs[0].cpu().numpy(), (1, 2, 0)), cv2.COLOR_BGR2RGB))
            plt.title('Original Image', fontdict=title_fontdict)
            plt.subplot(1, 4, 2)
            plt.axis("off")
            plt.imshow(np.transpose(rgb_amplitude_spectrum[0], (1, 2, 0)), cmap='gray')
            plt.title('Amplitude Spectrum', fontdict=title_fontdict)
            plt.subplot(1, 4, 3)
            plt.axis("off")
            plt.imshow(cv2.cvtColor(np.transpose(filtered_bd_image[0].cpu().numpy(), (1, 2, 0)), cv2.COLOR_BGR2RGB))
            plt.title(f'{args.cutoff_percentage}% Filtered Image', fontdict=title_fontdict)
            plt.savefig('dct.png')
            plt.subplot(1, 4, 4)
            plt.axis("off")
            plt.imshow(cv2.cvtColor(np.transpose((bd_imgs[0].cpu()-filtered_bd_image[0].cpu()).numpy(), (1, 2, 0)), cv2.COLOR_BGR2RGB))
            plt.title(f'ori-bd', fontdict=title_fontdict)
            plt.savefig('dct.png')
        pbar.close()

        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        df = pd.read_csv(args.result_csv_path)
        current_row = len(df)
        df.loc[current_row] = [formatted_time,args.filtered_type,args.cutoff_percentage,100 * ori_ACC / total,100 * ori_ASR / total,100 * ori_bd_robust / total,100 * ACC / total,100 * ASR / total,100 * bd_robust / total]
        df.to_csv(args.result_csv_path,index=False)
        
        with open(f'{args.result_save_path}/results_{args.filtered_type}_{args.cutoff_percentage}.txt', 'a+') as file:         
            print('##################################################')
            print('# Before Filter:')
            print('Accuracy of benigh inputs before Filter: %.3f %%' % (100 * ori_ACC / total))
            print('Attack success rate before Filter: %.3f %%' % (100 * ori_ASR / total))
            print('Backdoor Robustness after Filter: %.3f %%' % (100 * ori_bd_robust / total))
            print('\n# After Filter:\n')
            print('Accuracy of sanitized input after Filter: %.3f %%' % (100 * ACC / total))
            print('Atack Success rate after Filter: %.3f %%' % (100 * ASR / total))
            print('Backdoor Robustness after Filter: %.3f %%' % (100 * bd_robust / total))
            file.write('\n# After Filter:\n')
            file.write('Accuracy of sanitized input after Filter: %.3f %%\n' % (100 * ACC / total))
            file.write('Atack Success rate after Filter: %.3f %%\n' % (100 * ASR / total))
            file.write('Backdoor Robustness after Filter: %.3f %%\n' % (100 * bd_robust / total))
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

    args.result_save_path = f'./record/{args.result_file}/defense/filter'
    args.result_csv_path = f'./record/{args.result_file}/defense/filter/results.csv'

    if not os.path.exists(args.result_save_path):
        os.makedirs(args.result_save_path)

    if not os.path.exists(args.result_csv_path):
        data_columns = ['time','filter_type','cutoff','BA','ASR','RA','BA_filtered','ASR_filtered','RA_filtered']
        df = pd.DataFrame(columns=data_columns)
        df.to_csv(args.result_csv_path,index=False)
    
    with open(f'{args.result_save_path}/results_{args.filtered_type}_{args.cutoff_percentage}.txt', 'a+') as file:
        file.write('##################################################\n')
        file.write("==================Load result from: "+args.result_file+"========================\n")
        file.write(str(args))
    file.close()
            
    filter.defense(args.result_file)