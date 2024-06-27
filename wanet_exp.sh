#run
# python ./attack/sig.py \
#  --yaml_path ../config/attack/prototype/20-imagenet.yaml \
#  --save_folder_name sig_0_1

# shutdown
# shutdown –h now

# python ./attack/wanet.py \
#  --yaml_path ../config/attack/prototype/cifar10.yaml \
#  --save_folder_name cifar10_wanet_0


python ./attack/wanet.py \
 --yaml_path ../config/attack/prototype/imagenette-320.yaml \
 --save_folder_name imagenette-320_wanet_0