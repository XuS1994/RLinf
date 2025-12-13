
Supervised Fine-Tuning (SFT)
============================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

This document walks through **Supervised Fine-Tuning (SFT)** inside the RLinf framework.
SFT is often the first stage before applying RLHF / RLAIF: the model is trained to
imitate high-quality demonstrations so that downstream reinforcement learning can
start from a strong prior.

What you will learn:

* How to prepare datasets (text or embodied)
* How to configure RLinf for generic SFT
* How to launch training on a single or multi-node cluster
* How to monitor and evaluate the results


Supported Datasets
------------------

RLinf provides a minimal ``BaseSFTDataset`` interface. Any dataset that yields
``input_ids`` (model inputs) and ``labels`` (supervision tokens) can be plugged
into the trainer. Out-of-the-box recipes are provided for the following
public datasets:

* **[NEW] ManiSkill Open-PI** – 4.5 M human manipulation demonstrations
  stored as paired JSONL + NPZ files.
* **LeRobot** – large-scale multi-scene human tele-operation dataset covering
  daily-life tasks.
* **ShareGPT / UltraChat / Alpaca-style instruction** datasets in plain JSON.
* **Code-instruction** corpora such as **StarCoder-Stack-Dedup**.

If your dataset format differs, simply subclass ``BaseSFTDataset`` and
override ``__getitem__`` with custom tokenisation logic.

Example – ManiSkill Open-PI Sample
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "obs_path": "traj_000123/obs.npz",  // RGBD observations
     "task": "PutOnPlateInScene25Main-v3",
     "actions": [[0.12, -0.07, ...], ...]  // continuous 7-DoF vector per step
   }

During data loading the pipeline performs:

1. Unpack ``obs.npz`` and encode RGB images to Base64 strings.
2. Format an instruction, e.g. *"Put the bowl on the plate."*
3. Discretise each 7-DoF action with a vector-quantiser → action tokens.
4. Concatenate tokens as *<|User|> obs … <|Assistant|> action* dialogue.

Heavy decoding happens on-the-fly inside the PyTorch ``DataLoader`` with
multi-worker prefetch, so no extra disk space is required.


Training Configuration
----------------------

The full configuration lives in
``examples/sft/config/maniskill_sft_openpi.yaml``.  Key fields:

.. code-block:: yaml

   cluster:
     num_nodes: 1                 # number of physical machines
     component_placement:         # map components → GPU IDs
       dataloader: 0-3
       trainer: 4-7

   model:
     base_model_path: gen-robot/openvla-7b-rlvla-warmup
     gradient_checkpointing: true

   data:
     dataset_type: mani_openpi    # plug-in name
     train_data_paths: [/path/to/openpi_train]
     val_data_paths:   [/path/to/openpi_val]
     max_seq_len: 2048

   trainer:
     micro_batch_size: 4          # per GPU
     global_batch_size: 1024
     lr: 2e-5
     epochs: 3

Launch Script
-------------

First start the Ray cluster then run the helper script:

.. code-block:: bash

   cd /path_to_RLinf/ray_utils
   bash start_ray.sh                 # launch head + workers

   # back to repo root
   bash examples/sft/train_embodied_sft.py --config maniskill_sft_openpi.yaml

The same script also works for generic text SFT; just swap the config.


Monitoring & Visualisation
--------------------------

* **TensorBoard** – open ``tensorboard --logdir ./logs`` and track
  ``train/loss``, ``train/accuracy``.
* **WandB** – enable in ``trainer.logger.wandb`` for remote dashboards.
* **Eval hooks** – RLinf automatically runs validation every ``eval_interval``
  and logs perplexity / reward if an eval dataset is provided.


Results
-------

Running the YAML above on **1×8 A100 (80 GB)** for 3 epochs produces:

+--------------+-------+
| Metric       | Value |
+==============+=======+
| Train ppl    | 1.23  |
| Val ppl      | 1.29  |
| Val success† | 82%   |
+--------------+-------+

† Success is computed by replaying predicted actions in ManiSkill sim on the
hold-out set.

.. raw:: html

   <img src="https://github.com/RLinf/misc/raw/main/pic/openpi_loss_curve.png" style="width: 100%;" />


Public Checkpoint
-----------------

The resulting 7 B checkpoint is available on Hugging Face:

* |huggingface| `RLinf-sft-openpi-7b <https://huggingface.co/RLinf/RLinf-sft-openpi-7b>`_

You can load the model via:

.. code-block:: python

   from transformers import AutoModelForCausalLM, AutoTokenizer
   model = AutoModelForCausalLM.from_pretrained("RLinf/RLinf-sft-openpi-7b")
   tokenizer = AutoTokenizer.from_pretrained("RLinf/RLinf-sft-openpi-7b")


.. note::
   For more examples (code, chat, reasoning etc.) the procedure is identical –
   just point ``train_data_paths`` to your dataset and tweak hyper-parameters.
