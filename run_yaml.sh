#!/bin/bash

# 1. 核心路径配置
export REPO_PATH="/mnt/public/xttx/xusi/RLinf-v0.1"
export WORKDIR="$REPO_PATH/examples/embodiment"
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export SYNC_FLAG_FILE="$REPO_PATH/ray_utils/task_sync.txt"
export FORCE_REBUILD=1

# 2. 任务列表
#    格式: ENV_NAME MODEL_NAME YAML_ARG T_NODES T_STEPS T_SAVE
#    ENV_NAME: 环境名称 (maniskill_libero, behavior, isaaclab, metaworld, calvin等)
#    MODEL_NAME: 模型名称 (openvla, openvla-oft, openpi, gr00t, mlp等)
#    YAML_ARG: 配置文件名称
TASKS=(
    # 原有任务
    "maniskill_libero openpi maniskill_ppo_openpi 1 2 -1"  # v0.1 yaml不存在
    "maniskill_libero openpi maniskill_ppo_openpi_pi05 1 2 -1"  # v0.1 yaml不存在
    "maniskill_libero openvla maniskill_ppo_openvla 1 2 -1"
    "maniskill_libero openvla-oft maniskill_ppo_openvlaoft 1 2 -1"
    "maniskill_libero mlp maniskill_ppo_mlp 1 2 -1"
    "maniskill_libero openpi libero_goal_ppo_openpi 1 2 -1"
    "maniskill_libero openpi libero_goal_ppo_openpi_pi05 1 2 -1"
    "maniskill_libero gr00t libero_10_ppo_gr00t 1 2 -1"
    "behavior openpi behavior_ppo_openpi 1 2 -1"  # v0.1 yaml不存在
    "calvin openpi calvin_abc_d_ppo_openpi 1 2 -1"  # v0.1 yaml不存在
    "calvin openpi calvin_abcd_d_ppo_openpi_pi05 1 2 -1"  # v0.1 yaml不存在
    "robotwin openvla-oft robotwin_place_empty_cup_ppo_openvlaoft 1 2 -1"  # v0.1 yaml不存在
    "isaaclab gr00t isaaclab_franka_stack_cube_ppo_gr00t 1 2 -1"  # v0.1 yaml不存在
    "frankasim mlp frankasim_ppo_mlp 1 2 -1"  # v0.1 yaml不存在
    "maniskill_libero openpi gsenv_ppo_openpi_pi05 1 2 -1"  # v0.1 yaml不存在
    "maniskill_libero openpi maniskill_async_ppo_openpi 1 2 -1"  # v0.1 yaml不存在
    "maniskill_libero openpi maniskill_async_ppo_openpi_pi05 1 2 -1"  # v0.1 yaml不存在
    "maniskill_libero openvla maniskill_async_ppo_openvla 1 2 -1"  # v0.1 yaml不存在
    "maniskill_libero openvla-oft maniskill_async_ppo_openvlaoft 1 2 -1"  # v0.1 yaml不存在
    "maniskill_libero openpi libero_spatial_async_ppo_openpi 1 2 -1"  # v0.1 yaml不存在
    "maniskill_libero openpi libero_object_async_ppo_openpi_pi05 1 2 -1"  # v0.1 yaml不存在
    "maniskill_libero openpi realworld_45_ppo_openpi 1 2 -1"  # v0.1 yaml不存在
    "maniskill_libero openpi realworld_50_ppo_openpi_pi05 1 2 -1"  # v0.1 yaml不存在
    "maniskill_libero openvla maniskill_grpo_openvla 1 2 -1"
    "maniskill_libero openvla-oft maniskill_grpo_openvlaoft 1 2 -1"
    "maniskill_libero openpi libero_10_grpo_openpi 1 2 -1"
    "maniskill_libero openpi libero_spatial_grpo_openpi_pi05 1 2 -1"
    "maniskill_libero openvla-oft libero_10_grpo_openvlaoft 1 2 -1"
    "maniskill_libero mlp libero_spatial_0_grpo_mlp 1 2 -1"
    "robotwin openvla-oft robotwin_beat_block_hammer_grpo_openvlaoft 1 2 -1"  # v0.1 yaml不存在
    "wan openvla-oft wan_libero_goal_grpo_openvlaoft 1 2 -1"  # v0.1 yaml不存在
    "maniskill_libero mlp maniskill_sac_mlp 1 2 -1"  # v0.1 yaml不存在
    "frankasim mlp frankasim_sac_cnn_async 1 2 -1"  # v0.1 yaml不存在
    # 从workflow补充的任务
    # "maniskill_libero openvla maniskill_sac_mlp 1 2 -1"
    # "maniskill_libero openvla maniskill_sac_mlp_async 1 2 -1"
    # "maniskill_libero openvla maniskill_sac_flow_state 1 2 -1"
    # "maniskill_libero openvla realworld_dummy_sac_cnn 1 2 -1"
    # "frankasim openvla frankasim_ppo_mlp 1 2 -1"
    # "frankasim openvla frankasim_sac_cnn_async 1 2 -1"
    # "maniskill_libero openvla-oft libero_goal_grpo_openvlaoft 1 2 -1"
    # "behavior openvla-oft behavior_ppo_openvlaoft 1 2 -1"
    # "robotwin openvla-oft robotwin_grpo_openvlaoft 1 2 -1"
    # "maniskill_libero gr00t libero_spatial_ppo_gr00t 1 2 -1"
    # "isaaclab gr00t isaaclab_ppo_gr00t 1 2 -1"
    # "maniskill_libero openpi maniskill_ppo_openpi05 1 2 -1"
    # "maniskill_libero openpi libero_spatial_ppo_openpi 1 2 -1"
    # "maniskill_libero openpi libero_spatial_ppo_openpi05 1 2 -1"
    # "maniskill_libero openpi libero_spatial_dsrl_openpi 1 2 -1"
    # "maniskill_libero openpi maniskill_ppo_co_training_openpi_pi05 1 2 -1"
    # "metaworld openpi metaworld_50_ppo_openpi 1 2 -1"
    # "calvin openpi calvin_ppo_openpi 1 2 -1"
    # "maniskill_libero openpi robocasa_grpo_openpi 1 2 -1"
    # "maniskill_libero openvla-oft opensora_libero_spatial_grpo_openvlaoft 1 2 -1"
    # "maniskill_libero openvla-oft wan_libero_spatial_grpo_openvlaoft 1 2 -1"
)

RANK=${RANK:-0}
NUM_GPUS_PER_NODE=8

# 定义统一清理函数
function super_cleanup() {
    echo "[$(date +%T)] Performing aggressive cleanup..."
    # 尝试停止 ray，如果找不到命令则跳过
    command -v ray >/dev/null 2>&1 && ray stop --force || echo "Ray command not found, skipping ray stop"
    pkill -9 -u $(whoami) python >/dev/null 2>&1
    pkill -9 -u $(whoami) ray >/dev/null 2>&1
    rm -rf /dev/shm/ray/* 2>/dev/null
    sleep 3
}

# ---------------- RANK 分支逻辑 ----------------

if [ "$RANK" -eq 0 ]; then
    # ================= HEAD NODE 逻辑 =================
    
    # 启动前清理所有残留信号
    rm -f "$SYNC_FLAG_FILE"
    super_cleanup

    for TASK_STR in "${TASKS[@]}"; do
        read -r ENV_NAME MODEL_NAME YAML_ARG T_NODES T_STEPS T_SAVE <<< "$TASK_STR"
        
        echo "========================================================="
        echo "NEW TASK: $YAML_ARG | ENV: $ENV_NAME | MODEL: $MODEL_NAME"
        echo "========================================================="

        # 1. 确保 Worker 看到信号已消失（清理阶段）
        rm -f "$SYNC_FLAG_FILE"
        sleep 5 # 给共享文件系统同步时间

        # 2. 构建环境（如果.venv不存在或需要重建）
        cd "$REPO_PATH" || exit
        echo "Building environment: model=$MODEL_NAME, env=$ENV_NAME"
        
        # 设置环境变量（参考workflow中的设置）
        unset UV_DEFAULT_INDEX
        export UV_PATH=${UV_PATH:-/mnt/public/hao/models/.uv}
        export UV_LINK_MODE=${UV_LINK_MODE:-symlink}
        export UV_CACHE_DIR=${UV_CACHE_DIR:-/mnt/public/hao/models/.uv_cache}
        export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-/mnt/public/hao/models/.uv_python}
        
        # 根据环境设置特定路径
        case "$ENV_NAME" in
            maniskill_libero)
                export LIBERO_PATH=${LIBERO_PATH:-/mnt/public/hao/models/LIBERO}
                ;;
            behavior)
                export BEHAVIOR_PATH=${BEHAVIOR_PATH:-/mnt/public/hao/models/BEHAVIOR-1K}
                export ISAAC_SIM_WHEEL_PATH=${ISAAC_SIM_WHEEL_PATH:-/mnt/public/hao/models/isaac_sim_wheels}
                ;;
            isaaclab)
                export ISAAC_LAB_PATH=${ISAAC_LAB_PATH:-/mnt/public/hao/models/IsaacLab}
                export GR00T_PATH=${GR00T_PATH:-/mnt/public/hao/models/Isaac-GR00T/}
                ;;
            calvin)
                export CALVIN_PATH=${CALVIN_PATH:-/mnt/public/hao/models/calvin}
                ;;
            frankasim)
                export SERL_PATH=${SERL_PATH:-/mnt/public/hao/models/serl}
                ;;
            robotwin)
                export ROBOTWIN_PATH=${ROBOTWIN_PATH:-/mnt/public/hao/models/RoboTwin}
                ;;
        esac
        
        # 根据模型设置特定路径
        case "$MODEL_NAME" in
            gr00t)
                export GR00T_PATH=${GR00T_PATH:-/mnt/public/hao/models/Isaac-GR00T/}
                ;;
            openvla-oft)
                case "$ENV_NAME" in
                    opensora)
                        export OPENSORA_PATH=${OPENSORA_PATH:-/mnt/public/hao/models/opensora}
                        ;;
                    wan)
                        export WAN_PATH=${WAN_PATH:-/mnt/public/hao/models/wan}
                        ;;
                esac
                ;;
        esac
        
        # 构建环境（如果.venv不存在则构建，否则跳过）
        if [ ! -d ".venv" ] || [ "${FORCE_REBUILD:-0}" -eq 1 ]; then
            bash requirements/install.sh embodied --model "$MODEL_NAME" --env "$ENV_NAME"
            if [ $? -ne 0 ]; then
                echo "！！！CRITICAL ERROR: Environment build failed for model=$MODEL_NAME, env=$ENV_NAME"
                exit 1
            fi
        else
            echo "Using existing .venv, skipping build (set FORCE_REBUILD=1 to rebuild)"
        fi
        
        # 3. 激活环境
        if [ -f ".venv/bin/activate" ]; then
            source .venv/bin/activate
        else
            # 如果.venv不存在，尝试使用switch_env
            source switch_env "$MODEL_NAME" 2>/dev/null || {
                echo "Warning: Could not activate .venv or switch_env, continuing..."
            }
        fi
        
        super_cleanup # 环境加载后再清理，确保 ray stop 生效

        # 4. 写入新信号并启动 Head
        echo "$ENV_NAME" > "$SYNC_FLAG_FILE"
        echo "Head: Signal sent. Starting Ray Head..."
        
        export NODES=$T_NODES
        export STEPS=$T_STEPS
        export SAVE_INTER=$T_SAVE
        export TOKENIZERS_PARALLELISM=false

        bash ray_utils/start_ray.sh
        
        # 5. 等待集群就绪 (GPU总数，假设每节点4卡，2节点=8)
        TOTAL_GPUS=$(($T_NODES * $NUM_GPUS_PER_NODE))
        bash ray_utils/check_ray.sh "$TOTAL_GPUS"

        # 6. 执行任务（参考run_embodiment.sh的逻辑）
        cd "$WORKDIR" || exit
        echo "Executing training..."
        
        # 设置run_embodiment.sh需要的环境变量
        export MUJOCO_GL="egl"
        export PYOPENGL_PLATFORM="egl"
        export PYTHONPATH=${REPO_PATH}:$PYTHONPATH
        
        # 根据任务设置特殊环境变量（参考workflow）
        case "$YAML_ARG" in
            robotwin_*)
                export ROBOT_PLATFORM=${ROBOT_PLATFORM:-ALOHA}
                export ROBOTWIN_PATH=${ROBOTWIN_PATH:-/mnt/public/hao/models/RoboTwin}
                export PYTHONPATH=${ROBOTWIN_PATH}:$PYTHONPATH
                ;;
            behavior_*)
                export OMNIGIBSON_DATA_PATH=${OMNIGIBSON_DATA_PATH:-/mnt/public/hao/models/behavior-datasets}
                export ISAAC_PATH=${ISAAC_PATH:-/mnt/public/hao/models/isaac-sim}
                ;;
            isaaclab_*)
                # Isaac Lab环境变量在构建时已设置
                ;;
        esac
        
        # 执行训练脚本
        bash "${WORKDIR}/run_embodiment.sh" "$YAML_ARG" 2>&1 | tee "${YAML_ARG}_run.log"
        EXIT_CODE=${PIPESTATUS[0]}

        if [ $EXIT_CODE -ne 0 ]; then
            echo "！！！CRITICAL ERROR: $YAML_ARG failed with Code $EXIT_CODE"
            rm -f "$SYNC_FLAG_FILE"
            super_cleanup
            exit 1
        fi

        # 7. 任务成功，清除信号，准备下一轮
        echo "Task $YAML_ARG completed."
        rm -f "$SYNC_FLAG_FILE"
        super_cleanup
        sleep 10
    done

    echo "ALL TASKS COMPLETED!"

else
    # ================= WORKER NODE 逻辑 =================
    LAST_PROCESSED_ENV=""

    while true; do
        if [ ! -f "$SYNC_FLAG_FILE" ]; then
            echo "[$(date +%T)] Worker: Waiting for signal..."
            LAST_PROCESSED_ENV="" # 信号消失，重置记录
            sleep 5
            continue
        fi

        CURRENT_ENV=$(cat "$SYNC_FLAG_FILE" | tr -d '[:space:]')
        
        # 如果信号文件为空，或者环境还没变，则继续等待
        if [ -z "$CURRENT_ENV" ] || [ "$CURRENT_ENV" == "$LAST_PROCESSED_ENV" ]; then
            sleep 2
            continue
        fi

        echo "[$(date +%T)] Worker: New Signal [$CURRENT_ENV]. Initializing..."
        
        # 1. 切换环境并同步清理
        cd "$REPO_PATH" || exit
        source switch_env "$CURRENT_ENV"
        super_cleanup
        
        # 2. 启动 Ray 并加入集群
        echo "Worker: Joining Ray cluster with env $CURRENT_ENV..."
        bash ray_utils/start_ray.sh
        
        LAST_PROCESSED_ENV="$CURRENT_ENV"

        # 3. 阻塞等待任务结束（信号文件被 Head 删除）
        echo "Worker: Training in progress..."
        while [ -f "$SYNC_FLAG_FILE" ]; do
            # 检查收到的信号是否中途改变（虽然概率低）
            TMP_ENV=$(cat "$SYNC_FLAG_FILE" 2>/dev/null | tr -d '[:space:]')
            if [ "$TMP_ENV" != "$CURRENT_ENV" ] && [ -n "$TMP_ENV" ]; then
                echo "Worker: Signal changed mid-task! Re-initializing..."
                break
            fi
            sleep 10
        done
        
        echo "Worker: Task finished signal detected. Cleaning up..."
        super_cleanup
    done
fi
