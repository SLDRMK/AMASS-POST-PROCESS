#!/bin/bash

# IsaacGym环境设置脚本
# 解决编译和运行时的问题

# 设置Python库路径
export LD_LIBRARY_PATH=/home/sldrmk/miniconda3/envs/pbhc/lib:$LD_LIBRARY_PATH

# 使用系统编译器而不是conda编译器
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

# 添加系统头文件路径
export CPATH=/usr/include:$CPATH

# IsaacGym特定设置
export ISAACGYM_FORCE_CUDA=1
export ISAACGYM_FORCE_CUDA_ARCH=8.6
export TORCH_CUDA_ARCH_LIST="8.6"
export MAX_JOBS=1

# PyTorch扩展设置
export CMAKE_PREFIX_PATH=/home/sldrmk/miniconda3/envs/pbhc

# 禁用某些可能导致问题的功能
export ISAACGYM_DISABLE_CUDA_PREALLOCATE=1

echo "环境变量已设置完成！"
echo "现在可以运行IsaacGym相关程序了。" 