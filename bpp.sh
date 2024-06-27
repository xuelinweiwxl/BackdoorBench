#run
# python ./attack/bpp.py \
#  --yaml_path ../config/attack/prototype/20-imagenet.yaml \
#  --save_folder_name bpp_0_1

# shutdown
# shutdown –h now

# python ./attack/bpp.py \
#  --yaml_path ../config/attack/prototype/cifar10.yaml \
#  --save_folder_name cifar10_bpp_0


python ./attack/bpp.py \
 --yaml_path ../config/attack/prototype/imagenette-320.yaml \
 --save_folder_name imagenette-320_bpp_0