# #run
# python ./attack/sig.py \
#  --yaml_path ../config/attack/prototype/20-imagenet.yaml \
#  --save_folder_name sig_0_1

# # shutdown
# shutdown –h now

#run
# python ./attack/sig.py \
#  --yaml_path ../config/attack/prototype/cifar10.yaml \
#  --save_folder_name cifar10_sig_0

python ./attack/sig.py \
 --yaml_path ../config/attack/prototype/imagenette-320.yaml \
 --save_folder_name imagenette-320_sig_0
