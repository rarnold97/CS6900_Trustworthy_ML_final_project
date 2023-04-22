#!/usr/bin/env bash

### This File is intended to be sourced.

# Get the path of the current script
script_path="${BASH_SOURCE:-$0}"
# Resolve the script path to an absolute path
script_path="$(realpath "$script_path")"
# Extract the directory path from the script path
script_directory="$(dirname "$script_path")"

export CONFIG_FILE=${script_directory}/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml
export CKPT=${script_directory}/checkpoints/pointpillar_7728.pth
export BATCH_SIZE=4


test_pretrained_model()
{
    pushd ${script_directory}/OpenPCDet/tools
    python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --ckpt ${CKPT}
    popd
}