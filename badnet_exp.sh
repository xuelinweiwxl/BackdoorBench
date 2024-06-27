#run
###
 # @Author: Xuelin Wei
 # @Email: xuelinwei@seu.edu.cn
 # @Date: 2024-03-06 15:48:04
 # @LastEditTime: 2024-04-27 14:59:15
 # @LastEditors: xuelinwei xuelinwei@seu.edu.cn
 # @FilePath: /BackdoorBench/blended_exp.sh
### 
# python ./attack/blended.py \
#  --yaml_path ../config/attack/prototype/20-imagenet.yaml \
#  --attack_trigger_img_path ../resource/blended/hello_kitty.jpeg \
#  --save_folder_name test

# python ./attack/blended.py \
#  --yaml_path ../config/attack/prototype/cifar10.yaml \
#  --attack_trigger_img_path ../resource/blended/hello_kitty.jpeg \
#  --save_folder_name cifar10_blended_0

python ./attack/badnet.py \
 --yaml_path ../config/attack/prototype/imagenette-320.yaml \
 --save_folder_name imagenette-320_badnet_0
