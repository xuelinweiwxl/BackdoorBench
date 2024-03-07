python ./attack/ssba.py --yaml_path ../config/attack/prototype/20-imagenet.yaml --save_folder_name ssba_0_1

shutdown –h now

python ./attack/badnet.py --yaml_path ../config/attack/prototype/gtsrb.yaml --save_folder_name gtsrb_badnet_test
