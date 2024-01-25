#run
python ./attack/blended.py \
 --yaml_path ../config/attack/prototype/20-imagenet.yaml \
 --attack_trigger_img_path ../resource/blended/hello_kitty.jpeg \
 --save_folder_name blended_0_1

# shutdown
shutdown –h now