#!/bin/bash

# 定义要执行的命令
command="python ./defense/freq_test_wxl.py --replace_mode 0 --result_file"

# percentage list
b_list=("1" "0.8" "0.6" "0.4" "0.2" "0") 

# filter type
a_list=("0.15" "0.12" "0.1" "0.09" "0.05" "0.01")

# 定义要遍历的model参数列表
file_list=("badnet_0_2"
            "blended_0_2"
            "bpp_0_1"
            "input-aware_0_1"
            "sig_0_1"
            "ssba_0_1"
            "trojannn_0_1"
            "wanet_0_1")

# file
for file in "${file_list[@]}"; do
    for alpha in "${a_list[@]}"; do
        for beta in "${b_list[@]}"; do
            # 构建完整的命令
            full_command="$command $file --alpha $alpha --beta $beta"

            # 执行命令
            echo "Executing command: $full_command"
            $full_command
        done
    done
done