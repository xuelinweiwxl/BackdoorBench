#!/bin/bash

# 定义要执行的命令
command="python ./defense/freq3_test_wxl.py --result_file"

# prototype file location
yaml_path="./config/defense/freq_test_wxl/imagenette-320.yaml"

# 定义要遍历的model参数列表
file_list=("imagenette-320_badnet_0"
            "imagenette-320_blended_0"
            "imagenette-320_inputaware_0"
            "imagenette-320_sig_0"
            "imagenette-320_trojannn_0"
            "imagenette-320_wanet_0")

# file
for file in "${file_list[@]}"; do
    # 构建完整的命令
    full_command="$command $file --yaml_path $yaml_path"

    # 执行命令
    echo "Executing command: $full_command"
    $full_command
done