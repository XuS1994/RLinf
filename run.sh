conda activate /mnt/public/mjwei/conda_envs/zqlenv_wmj_0729

bash examples/embodiment/run_pi0.sh
bash examples/embodiment/run_pi0_h100.sh

cd /mnt/public/chenkang/eai
cd /mnt/public/chenkang/megatron-infinigence-rl-chenk/
source /mnt/public/chenkang/miniconda/envs/openpi/bin/activate
tensorboard --host 0.0.0.0 --logdir /mnt/public/liuzhihao/RLinf_0828/logs
tensorboard --host 0.0.0.0 --logdir /mnt/mnt/public/liuzhihao/RLinf_0828/logs

tensorboard --host 0.0.0.0 --logdir /mnt/public/chenkang/megatron-infinigence-rl-chenk/logs/

tensorboard --host 0.0.0.0 --logdir outputs/rl


cd /mnt/public/chenkang/eai/
source /mnt/public/chenkang/miniconda/envs/openpi/bin/activate
tensorboard --host 0.0.0.0 --logdir outputs/rl

kill -9 $(ps aux | grep 'train_embodied_agent.py' | grep -v grep | awk '{print $2}')

conda activate vla
cd /mnt/public/chenkang/megatron-infinigence-rl-chenk
nohup bash examples/embodiment/run_pi0_h100_few1.sh > run_pi0_h100_few1.log 2>&1 &
nohup bash examples/embodiment/run_pi0_h100_few1_big_lr.sh > run_pi0_h100_few1_big_lr.log 2>&1 &

nohup bash examples/embodiment/run_pi0_h100_few1_big_lr2.sh > run_pi0_h100_few1_big_lr2.log 2>&1 &
nohup bash examples/embodiment/run_pi0_h100_few1_big_lr3.sh > run_pi0_h100_few1_big_lr3.log 2>&1 &


nohup bash examples/embodiment/run_pi0_h100_few3.sh > run_pi0_h100_few3.log 2>&1 &

# download from BEIJING I 
rsync -av -P --partial-dir=.rsync-part -e "ssh -J ssh-jumper.cloud.infini-ai.com" \
--exclude='pretrained_model/' \
--exclude='outputs/' \
--exclude='RLinf/' \
--exclude='logs/' \
--exclude='.git/' \
root@aic-dbdz3pprd2g54crp:/mnt/public/liuzhihao/megatron-infinigence-rl-chenk \
/data/home/chenkang/eai/

# transfer to BEIJING J 
rsync -av -P --partial-dir=.rsync-part -e "ssh -J ssh-jumper.cloud.infini-ai.com" \
--exclude='pretrained_model/' \
--exclude='outputs/' \
--exclude='RLinf/' \
--exclude='logs/' \
--exclude='.git/' \
/data/home/chenkang/eai/megatron-infinigence-rl-chenk \
root@aic-dbd2ywp3oovhhb53:/mnt/public/chenkang/

cd /mnt/mnt/public/liuzhihao/RLinf_0828/
tmux

# ! test1
bash examples/embodiment/J2_run_pi0_grpo_8h100.sh J2_libero_spatial_grpo_pi0_lr2_bs64_grad_1_nofilter_normadv_seed42
# test2 - 重新包裹FSDP + filter + minibs128
bash examples/embodiment/J2_run_pi0_grpo_8h100.sh J2_spatial_grpo_lr2_bs128_grad_1_filter_normadv
# test3 - tensorboard 第四个结果
# 用内部函数，相当于单卡
# ! test4 - tensorboard 第三个结果
# 用forward 且全部是True

# test5 - 重新包裹FSDP 
bash examples/embodiment/J2_run_pi0_grpo_8h100.sh J2_libero_spatial_grpo_pi0_lr2_bs64_grad_1_nofilter_normadv_seed42
# test6 - 重新包裹FSDP + filter 
bash examples/embodiment/J2_run_pi0_grpo_8h100.sh J2_libero_spatial_grpo_pi0_lr2_bs64_grad_1_filter_normadv_seed42
# test7 - 重新包裹FSDP minibs128
bash examples/embodiment/J2_run_pi0_grpo_8h100.sh J2_spatial_grpo_lr2_bs128_grad_1_nofilter_normadv
# test8 - 重新包裹FSDP minibs128 + noise07
bash examples/embodiment/J2_run_pi0_grpo_8h100.sh J2_spatial_grpo_lr2_bs128_grad_1_nofilter_normadv_noise07



