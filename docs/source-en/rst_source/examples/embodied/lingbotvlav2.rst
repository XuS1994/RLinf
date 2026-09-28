PPO on LingbotVLA 2.0
=====================

Fine-tune LingbotVLA 2.0 on RoboTwin ``click_bell`` with PPO, then evaluate the
saved policy. The default configuration requires four GPUs with 48 GB each.

Installation
------------

1. Clone the RLinf Repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Clone RLinf and set the repository path:

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf
   export RLINF_PATH=$(pwd)

2. Install Dependencies
~~~~~~~~~~~~~~~~~~~~~~~

Create a dedicated environment for LingbotVLA 2.0:

.. code-block:: bash

   bash requirements/install.sh embodied --model lingbotvlav2 --env robotwin \
     --venv .venv-lingbotvlav2 --torch 2.9.0 --transformers 4.57.6
   source .venv-lingbotvlav2/bin/activate

To build a Docker image with the same dependencies:

.. code-block:: bash

   docker buildx build --load -f docker/Dockerfile \
     --build-arg BUILD_TARGET=embodied-robotwin-lingbotvlav2 \
     -t rlinf:robotwin-lingbotvlav2 .

RoboTwin Repository and Assets
------------------------------

Clone the RLinf-compatible RoboTwin branch and download its assets:

.. code-block:: bash

   export ROBOTWIN_PATH="$RLINF_PATH/../RoboTwin"
   git clone --branch RLinf_support https://github.com/RoboTwin-Platform/RoboTwin.git "$ROBOTWIN_PATH"
   (cd "$ROBOTWIN_PATH" && bash script/_download_assets.sh)
   export ROBOTWIN_ASSETS_PATH="$ROBOTWIN_PATH"

Download the Model
------------------

Download the V2 RoboTwin checkpoint and Qwen3-VL backbone:

.. code-block:: bash

   export MODEL_ROOT="$RLINF_PATH/../models"
   hf download robbyant/lingbot-vla-v2-6b-robotwin \
     --local-dir "$MODEL_ROOT/lingbot-vla-v2-6b-robotwin"
   hf download Qwen/Qwen3-VL-4B-Instruct \
     --local-dir "$MODEL_ROOT/Qwen3-VL-4B-Instruct"

After downloading, fill in the empty fields in
``examples/embodiment/config/model/lingbotvlav2.yaml`` with your absolute paths.
Training and standalone evaluation share this file:

.. code-block:: yaml

   model_path: /path/to/models/lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt
   tokenizer_path: /path/to/models/Qwen3-VL-4B-Instruct
   lingbotvlav2:
     training_config_path: /path/to/models/lingbot-vla-v2-6b-robotwin/lingbotvla_cli.yaml
     robot_config_path: /path/to/RLinf/.venv-lingbotvlav2/lingbot-vla-v2/configs/robot_configs/robotwin.yaml
     stats_path: /path/to/RLinf/.venv-lingbotvlav2/lingbot-vla-v2/assets/norm_stats/robotwin.json

Run It
------

From the RLinf repository, launch the PPO configuration:

.. code-block:: bash

   cd "$RLINF_PATH"
   bash examples/embodiment/run_embodiment.sh robotwin_click_bell_ppo_lingbotvlav2 ALOHA

To resume, set ``runner.resume_dir`` in
``examples/embodiment/config/robotwin_click_bell_ppo_lingbotvlav2.yaml`` to a saved
checkpoint directory.
Monitor ``env/success_once`` in TensorBoard.

Evaluation
----------

Evaluate the trained actor checkpoint:

.. code-block:: bash

   export ROBOT_PLATFORM=ALOHA
   export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
   bash evaluations/run_eval.sh robotwin_click_bell_lingbotvlav2_eval \
     runner.ckpt_path=/path/to/checkpoints/global_step_N/actor/model_state_dict/full_weights.pt

Omit ``runner.ckpt_path`` to evaluate the original model. See
:doc:`../../evaluations/index` for evaluation options.
