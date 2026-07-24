<div align="center">

# SkillSpotter: Pose-Aware Multi-View Skilled Action Detection and Grading in Ego-Exo Videos

[Björn Braun](https://bjoernbraun.com) &nbsp;·&nbsp; [Christian Holz](https://siplab.org)

[Sensing, Interaction & Perception Lab](https://siplab.org), Department of Computer Science, ETH Zürich

[![Project Page](https://img.shields.io/badge/Project-Page-1f72b1)](https://siplab.org/projects/SkillSpotter)
[![Paper](https://img.shields.io/badge/Paper-ECCV%202026-b31b1b)](https://static.siplab.org/papers/eccv2026-skillspotter.pdf)
[![arXiv](https://img.shields.io/badge/arXiv-2606.31127-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2606.31127)

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

Timestamp-level skill assessment on Ego-Exo4D demonstration proficiency. We report
class-specific mAP (mAP<sub>S</sub>), class-agnostic mAP (mAP<sub>A</sub>), balanced accuracy (BA),
and macro-F1 across the three view settings, averaged over matching radii {0.25, 0.5, 1.0}s.

<table>
<thead>
<tr>
<th rowspan="2" align="left">Model</th>
<th colspan="4">Ego</th>
<th colspan="4">Exos</th>
<th colspan="4">Ego + Exos</th>
</tr>
<tr>
<th>mAP<sub>S</sub></th><th>mAP<sub>A</sub></th><th>BA</th><th>F1</th>
<th>mAP<sub>S</sub></th><th>mAP<sub>A</sub></th><th>BA</th><th>F1</th>
<th>mAP<sub>S</sub></th><th>mAP<sub>A</sub></th><th>BA</th><th>F1</th>
</tr>
</thead>
<tbody>
<tr><td align="left">Random</td><td>0.73</td><td>1.49</td><td>50.90</td><td>50.03</td><td>0.70</td><td>1.47</td><td>50.44</td><td>49.59</td><td>0.70</td><td>1.46</td><td>50.15</td><td>49.39</td></tr>
<tr><td align="left">Uniform tips</td><td>0.71</td><td>1.49</td><td>50.00</td><td>27.15</td><td>0.71</td><td>1.47</td><td>50.00</td><td>27.15</td><td>0.72</td><td>1.46</td><td>50.00</td><td>27.15</td></tr>
<tr><td align="left">Uniform good</td><td>0.70</td><td>1.52</td><td>50.00</td><td>38.55</td><td>0.68</td><td>1.47</td><td>50.00</td><td>38.55</td><td>0.67</td><td>1.46</td><td>50.00</td><td>38.55</td></tr>
<tr><td align="left">Baseline<sup>†</sup></td><td>3.27</td><td>–</td><td>–</td><td>–</td><td>3.84</td><td>–</td><td>–</td><td>–</td><td>3.57</td><td>–</td><td>–</td><td>–</td></tr>
<tr><td align="left">VideoMambaSuite</td><td>7.63</td><td>8.65</td><td>49.51</td><td>49.35</td><td>7.06</td><td>8.17</td><td>46.03</td><td>43.88</td><td>3.69</td><td>3.91</td><td>52.70</td><td>51.96</td></tr>
<tr><td align="left">TadTR</td><td>7.79</td><td>10.79</td><td>49.68</td><td>33.62</td><td>6.37</td><td>10.48</td><td>52.31</td><td>34.63</td><td>4.12</td><td>6.74</td><td>52.81</td><td>48.23</td></tr>
<tr><td align="left">DyFADet</td><td>10.09</td><td>12.53</td><td>48.89</td><td>46.75</td><td>3.57</td><td>4.48</td><td>49.52</td><td>47.72</td><td>3.18</td><td>5.64</td><td>47.63</td><td>35.45</td></tr>
<tr><td align="left">TriDet</td><td>10.35</td><td>14.79</td><td>49.17</td><td>40.10</td><td>8.99</td><td>12.16</td><td>50.06</td><td>36.78</td><td>8.23</td><td>11.78</td><td>48.92</td><td>48.77</td></tr>
<tr><td align="left">CausalTAD</td><td>11.42</td><td>16.07</td><td>52.86</td><td>52.07</td><td>11.82</td><td>14.05</td><td>50.78</td><td>50.41</td><td>13.16</td><td>17.21</td><td>54.98</td><td>54.95</td></tr>
<tr><td align="left">TemporalMaxer</td><td>12.34</td><td>16.69</td><td>54.34</td><td>53.90</td><td>10.38</td><td>15.17</td><td>52.18</td><td>52.16</td><td>11.27</td><td>15.75</td><td>50.09</td><td>49.58</td></tr>
<tr><td align="left">ActionFormer</td><td>12.40</td><td>17.11</td><td>55.99</td><td>55.91</td><td>13.18</td><td>18.25</td><td>55.03</td><td>55.03</td><td>13.82</td><td>17.85</td><td>50.34</td><td>50.17</td></tr>
<tr><td align="left"><b>SkillSpotter</b><sup>*</sup></td><td><b>21.82</b></td><td><b>27.89</b></td><td><b>60.40</b></td><td><b>60.02</b></td><td><b>21.12</b></td><td><b>27.47</b></td><td><b>60.59</b></td><td><b>60.55</b></td><td><b>21.34</b></td><td><b>28.01</b></td><td><b>60.39</b></td><td><b>59.37</b></td></tr>
<tr><td align="left">Δ best baseline</td><td>+9.42</td><td>+10.78</td><td>+4.41</td><td>+4.11</td><td>+7.94</td><td>+9.22</td><td>+5.56</td><td>+5.52</td><td>+7.52</td><td>+10.16</td><td>+5.41</td><td>+4.42</td></tr>
</tbody>
</table>

<sup>†</sup> Original Ego-Exo4D benchmark result, for which only mAP<sub>S</sub> was reported. &nbsp;
<sup>*</sup> Uses cross-view attention for Ego+Exos. All re-implemented baselines use Soft-NMS.

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
@article{braun2026skillspotter,
  title   = {SkillSpotter: Pose-Aware Multi-View Skilled Action Detection and Grading in Ego-Exo Videos},
  author  = {Braun, Bj{\"o}rn and Holz, Christian},
  journal = {arXiv preprint arXiv:2606.31127},
  year    = {2026}
}
```

## Acknowledgements

This codebase builds on [ActionFormer](https://github.com/happyharrycn/actionformer_release) and
the [OpenTAD](https://github.com/sming256/OpenTAD) toolbox, and uses the
[Ego-Exo4D](https://ego-exo4d-data.org) dataset and its ego body-pose baseline. We thank the
authors for releasing their code and data.
