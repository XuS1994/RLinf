#!/bin/bash

# 测试Pi0模型使用FSDP wrapper的gradnorm一致性

echo "开始测试Pi0模型 + FSDP wrapper..."

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_ADDR=localhost
export MASTER_PORT=12355

# 运行测试
torchrun --nproc_per_node=2 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT test.py

echo "测试完成！"
