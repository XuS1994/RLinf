#! /bin/bash
export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"

export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH
# NOTE: set LIBERO_REPO_PATH to the path of the LIBERO repo
export LIBERO_REPO_PATH="/workspace/libero"
# NOTE: set LIBERO_CONFIG_PATH for libero/libero/__init__.py
export LIBERO_CONFIG_PATH=${LIBERO_REPO_PATH}
echo "LIBERO_CONFIG_PATH: ${LIBERO_CONFIG_PATH}"

export PYTHONPATH=${LIBERO_REPO_PATH}:$PYTHONPATH
echo "PYTHONPATH: ${PYTHONPATH}"
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1

if [ -z "$1" ]; then
    CONFIG_NAME=""
else
    CONFIG_NAME=$1
fi

LOG_DIR="${REPO_PATH}/logs/${CONFIG_NAME}_$(date +'%Y%m%d-%H:%M:%S')" #/$(date +'%Y%m%d-%H:%M:%S')"
MEGA_LOG_FILE="${LOG_DIR}/run_ppo.log"
mkdir -p "${LOG_DIR}"
CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR}"
echo ${CMD}
${CMD} 2>&1 | tee ${MEGA_LOG_FILE}
