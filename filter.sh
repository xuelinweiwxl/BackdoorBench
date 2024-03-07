#!/bin/bash

# 定义要执行的命令
command="python ./defense/filter.py --result_file"

# percentage list
p_list=("100" "98" "97" "95" "94" "93" "92" "91" "90")

# filter type
f_list=("high_ideal" "gaussian" "specific")

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
    for filter in "${f_list[@]}"; do
        for percentage in "${p_list[@]}"; do
            # 构建完整的命令
            full_command="$command $file --cutoff_percentage $percentage --filtered_type $filter"

            # 执行命令
            echo "Executing command: $full_command"
            $full_command
        done
    done
done