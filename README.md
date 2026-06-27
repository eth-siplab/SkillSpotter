<div align="center">

# SkillSpotter: Pose-Aware Multi-View Skilled Action Detection and Grading in Ego-Exo Videos

[Björn Braun](https://bjoernbraun.com) &nbsp;·&nbsp; [Christian Holz](https://siplab.org)

Sensing, Interaction & Perception Lab, Department of Computer Science, ETH Zürich

[![Project Page](https://img.shields.io/badge/Project-Page-1f72b1)](https://siplab.org/projects/SkillSpotter)
[![Paper](https://img.shields.io/badge/Paper-ECCV%202026-b31b1b)](https://siplab.org/projects/SkillSpotter)

</div>

---

## Overview

To enable personalized, real-time coaching with AR glasses or fixed camera setups in domains
such as sports, cooking, or music, a system must understand not just *what* a person does, but
*how well* they execute an activity. This requires simultaneously **detecting** individual skilled
actions in untrimmed ego-exo video and **grading** each as correct or needing improvement.

**SkillSpotter** is a pose-aware, multi-view framework for timestamp-level skill detection and
grading on the Ego-Exo4D *proficiency demonstration* benchmark. It builds on an ActionFormer
(ConvTransformer) backbone and adds three task-specific modules:

1. **Adaptive temporal suppression** — learns activity-specific suppression radii instead of a
   fixed-radius NMS, to handle the large variation in skilled-action density across activities.
2. **Gated 3D body-pose fusion** — incorporates body kinematics as a complementary signal to
   visual features for assessing execution quality.
3. **Bidirectional cross-view attention** — combines ego and exo views to prevent the grading
   collapse observed when naively concatenating the two streams.

On Ego-Exo4D, SkillSpotter improves class-specific mAP from 12.40 to 21.82 (+76%) and balanced
accuracy from 55.99% to 60.40% over the best baseline, with consistent gains across all view
settings and activities.

## Installation

```bash
git clone https://github.com/eth-siplab/SkillSpotter.git
cd SkillSpotter

conda create -n skillspotter python=3.9
conda activate skillspotter
pip install -r requirements.txt
```

The core training/evaluation pipeline runs on pre-extracted [Omnivore](https://github.com/facebookresearch/omnivore)
features and does not require any compiled extensions. The optional dependencies in
`requirements.txt` (pose prediction, raw-video clip extraction, text evaluation) can be installed
on demand.

## Data Preparation

SkillSpotter is trained and evaluated on the [Ego-Exo4D](https://ego-exo4d-data.org) *proficiency
demonstration* task. The pipeline runs end-to-end through the scripts in `tools/`:

**1. Download Ego-Exo4D.** Follow the [official instructions](https://docs.ego-exo4d-data.org) to
download the proficiency-demonstration takes, annotations, and the Aria camera trajectories
(`ego_pose`) used as the ground-truth pose source.

**2. Extract Omnivore features and build the dataset manifest.**

```bash
python tools/create_egoexo4d_demonstrator_omnivore.py \
    --base_dir /path/to/EgoExo4D \
    --omnivore_out_dir /path/to/EgoExo4D/features/omnivore_features \
    --out_json /path/to/EgoExo4D/annotations/egoexo4d.json \
    --views ego exo1 exo2 exo3 exo4
```

This writes a single `egoexo4d.json` manifest (in Ego-Exo4D "database" format, with both ego and
exo views) and per-view `.npy` Omnivore feature clips.

**3. Predict 3D body pose** (train on GT pose, test on predicted pose — see note below).

List the take UIDs that need pose prediction, then run the ego and exo predictors:

```bash
python tools/get_uids_pose_prediction.py        # writes needed_take_uids.txt

# Ego pose: official Ego-Exo4D baseline from Aria camera trajectories.
# Requires: git clone https://github.com/EGO4D/ego-exo4d-egopose.git
python tools/predict_ego_pose.py \
    --baseline_repo /path/to/ego-exo4d-egopose/bodypose \
    --checkpoint    /path/to/ego-exo4d-egopose/bodypose/.../100000_G.pth \
    --output_root   /path/to/EgoExo4D/predicted_ego_pose \
    --device cuda:0

# Exo pose: YOLO person detection + ViTPose 2D keypoints + multi-view triangulation.
python tools/predict_exo_pose.py \
    --egoexo_root /path/to/EgoExo4D \
    --output_root /path/to/EgoExo4D/predicted_exo_pose \
    --device cuda:0
```

Both predictors write pose in the same JSON format as the Ego-Exo4D `ego_pose` annotations, so
they are drop-in replacements selected via the `pose_root` config key.

> **Note — no ground-truth leakage.** We **train with the ground-truth pose** (`pose_root` →
> `annotations/ego_pose`) and **evaluate with our predicted pose** (`pose_root` →
> `predicted_ego_pose`). Set `pose_root` accordingly for the training and evaluation runs.

**4. Set the data paths in the config.** Open
[`configs/egoexo4d_skillspotter_omnivore.yaml`](configs/egoexo4d_skillspotter_omnivore.yaml) and
update the following keys to your machine (these are the only paths you need to change for
train/eval):

| Key | Meaning |
|-----|---------|
| `dataset.json_file`   | Path to `egoexo4d.json` (step 2) |
| `dataset.feat_folder` | Omnivore feature directory (step 2) |
| `dataset.pose_root`   | GT pose for training, predicted pose for evaluation (step 3) |
| `output_folder`       | Where checkpoints and logs are written |

Most `tools/` scripts expose their input/output paths as command-line flags (with defaults you can
override). Two helper scripts — `tools/get_uids_pose_prediction.py` and
`tools/convert_egoexo4d_trainvaltest.py` — instead define their paths as constants at the top of
the file; edit those few lines before running them.

## Training

```bash
python train.py configs/egoexo4d_skillspotter_omnivore.yaml
```

The same config drives every experiment via three module toggles and the view setting:

| Setting | Config |
|---------|--------|
| Adaptive temporal suppression | `model.learnable_suppression: True/False` |
| Gated 3D body-pose fusion      | `model.use_pose: True/False` |
| Bidirectional cross-view attn  | `dataset.cross_view: True/False` |
| View setting                   | `dataset.egoexo_type: ego / exo / both` |

GPUs are selected with the `devices` field in the config (e.g. `['cuda:0']` for single-GPU, or a
list for multi-GPU DDP). Training the full Ego+Exos model takes under 15 minutes on a single GPU.

## Evaluation

```bash
python eval.py configs/egoexo4d_skillspotter_omnivore.yaml <checkpoint_folder> --use_best_epoch true
```

`<checkpoint_folder>` is the run directory created under `output_folder` during training. The
evaluator reports class-agnostic mAP (mAP_A, detection), class-specific mAP (mAP_S), balanced
accuracy (BA), and macro-F1, averaged over matching radii {0.25, 0.5, 1.0}s. Remember to point
`dataset.pose_root` at the **predicted** pose for evaluation.

## Results

Timestamp-level skill assessment on Ego-Exo4D demonstration proficiency (mAP_S / BA):

| View setting | mAP_S | Balanced Acc. |
|--------------|:-----:|:-------------:|
| Ego          | 21.82 | 60.40 |
| Exos         | 21.12 | 60.59 |
| Ego + Exos   | 21.34 | 60.39 |

## Repository Structure

```
configs/    YAML experiment configs
libs/
  core/         config loading
  datasets/     Ego-Exo4D Omnivore + pose dataset
  modeling/     SkillSpotter model (skillspotter.py) and shared building blocks
  evaluating/   timestamp detection + grading evaluator
  utils/        training loop, schedulers, NMS, post-processing
tools/      dataset conversion, Omnivore feature extraction, ego/exo pose prediction
train.py    training entry point
eval.py     evaluation entry point
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{braun2026skillspotter,
  title     = {SkillSpotter: Pose-Aware Multi-View Skilled Action Detection and Grading in Ego-Exo Videos},
  author    = {Braun, Bj{\"o}rn and Holz, Christian},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## Acknowledgements

This codebase builds on [ActionFormer](https://github.com/happyharrycn/actionformer_release) and
the [OpenTAD](https://github.com/sming256/OpenTAD) toolbox, and uses the
[Ego-Exo4D](https://ego-exo4d-data.org) dataset and its ego body-pose baseline. We thank the
authors for releasing their code and data.
