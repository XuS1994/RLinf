LingbotVLA 2.0 的 PPO 训练
==========================

使用 PPO 在 RoboTwin ``click_bell`` 任务上微调 LingbotVLA 2.0，再评估训练后的 policy。默认配置需要 4 张 48 GB GPU。

安装
----

1. 克隆 RLinf 仓库
~~~~~~~~~~~~~~~~~~

克隆 RLinf 并设置仓库路径：

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf
   export RLINF_PATH=$(pwd)

2. 安装依赖
~~~~~~~~~~~

为 LingbotVLA 2.0 创建独立环境：

.. code-block:: bash

   bash requirements/install.sh embodied --model lingbotvlav2 --env robotwin \
     --venv .venv-lingbotvlav2 --torch 2.9.0 --transformers 4.57.6
   source .venv-lingbotvlav2/bin/activate
   export LINGBOT_VLA_V2_PATH="$VIRTUAL_ENV/lingbot-vla-v2"

也可以构建包含相同依赖的 Docker 镜像：

.. code-block:: bash

   docker buildx build --load -f docker/Dockerfile \
     --build-arg BUILD_TARGET=embodied-robotwin-lingbotvlav2 \
     -t rlinf:robotwin-lingbotvlav2 .

RoboTwin 仓库与资产
-------------------

克隆兼容 RLinf 的 RoboTwin 分支并下载资产：

.. code-block:: bash

   export ROBOTWIN_PATH="$RLINF_PATH/../RoboTwin"
   git clone --branch RLinf_support https://github.com/RoboTwin-Platform/RoboTwin.git "$ROBOTWIN_PATH"
   (cd "$ROBOTWIN_PATH" && bash script/_download_assets.sh)
   export ROBOTWIN_ASSETS_PATH="$ROBOTWIN_PATH"

下载模型
--------

下载 V2 RoboTwin checkpoint 和 Qwen3-VL 底座模型：

.. code-block:: bash

   export MODEL_ROOT="$RLINF_PATH/../models"
   hf download robbyant/lingbot-vla-v2-6b-robotwin \
     --local-dir "$MODEL_ROOT/lingbot-vla-v2-6b-robotwin"
   hf download Qwen/Qwen3-VL-4B-Instruct \
     --local-dir "$MODEL_ROOT/Qwen3-VL-4B-Instruct"

   export LINGBOT_VLA_V2_CKPT="$MODEL_ROOT/lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt"
   export LINGBOT_VLA_V2_TRAIN_CONFIG="$MODEL_ROOT/lingbot-vla-v2-6b-robotwin/lingbotvla_cli.yaml"
   export QWEN3_VL_PATH="$MODEL_ROOT/Qwen3-VL-4B-Instruct"

运行
----

在 RLinf 仓库中启动 PPO 训练：

.. code-block:: bash

   cd "$RLINF_PATH"
   bash examples/embodiment/run_embodiment.sh robotwin_click_bell_ppo_lingbotvlav2 ALOHA

续训时追加 ``runner.resume_dir=/path/to/checkpoints/global_step_N``。在 TensorBoard 中观察 ``env/success_once``。

评估
----

加载训练后的 actor checkpoint 进行评估：

.. code-block:: bash

   export ROBOT_PLATFORM=ALOHA
   export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
   bash evaluations/run_eval.sh robotwin_click_bell_lingbotvlav2_eval \
     runner.ckpt_path=/path/to/checkpoints/global_step_N/actor/model_state_dict/full_weights.pt

省略 ``runner.ckpt_path`` 可评估原始模型。其他选项见 :doc:`../../evaluations/index`。
