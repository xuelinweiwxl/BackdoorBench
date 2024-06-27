# #run
###
 # @Author: Xuelin Wei
 # @Email: xuelinwei@seu.edu.cn
 # @Date: 2024-03-06 15:48:04
 # @LastEditTime: 2024-04-27 19:31:03
 # @LastEditors: xuelinwei xuelinwei@seu.edu.cn
 # @FilePath: /BackdoorBench/trojannn_exp.sh
### 
# python ./attack/trojannn.py \
#   --yaml_path ../config/attack/prototype/20-imagenet.yaml \
#   --save_folder_name trojannn_0_1

# # shutdown
# shutdown –h now

#run
# python ./attack/prototype.py \
#   --yaml_path ../config/attack/prototype/cifar10.yaml \
#   --save_folder_name cifar10_prototype

python ./attack/prototype.py \
  --yaml_path ../config/attack/prototype/imagenette-320.yaml \
  --save_folder_name imagenette-320_prototype