import glob
import os
import json
import random

import torch
import numpy as np

from generate_canonical_descriptions import normalize_text

"""
Directory structure before processing:

This folder
│  convert_egoexo4d_trainval.py
│  ... 
│
└───features
│    └───omnivore_video_swinl
│
└───annotations
│    └───proficiency_demonstration_train.json
│    └───proficiency_demonstration_val.json
│  ...
"""

base_dir = '/path/to/EgoExo4D'

# full-video features downloaded from Ego4D website
omnivore_dir = base_dir + '/features/omnivore_video'

# annotation files downloaded from EgoExo4D website
train_annot_path = base_dir + '/annotations/proficiency_demonstration_train.json'
val_annot_path = base_dir + '/annotations/proficiency_demonstration_val.json'

# where to save the processed features
omnivore_out_dir = base_dir + '/features/omnivore_features'
os.makedirs(omnivore_out_dir, exist_ok=True)

# where to save the processed annotations
annot_out_path = base_dir + '/annotations/egoexo4d.json'

# load takes metadata
with open(base_dir + '/takes.json', 'r') as f:
    takes_metadata_temp = json.load(f)
takes_metadata = {take['take_uid']: take for take in takes_metadata_temp}

# clip size / stride in feature extraction
clip_size = 32
stride = 16

with open(train_annot_path, 'r') as f:
    original_train_takes = json.load(f)['annotations']
with open(val_annot_path, 'r') as f:
    original_val_takes = json.load(f)['annotations']

# 1. Convert original Validation set to Test set
for take in original_val_takes:
    take['split'] = 'test'

# 2. Split original Training set into Train (90%) and Validation (10%)
# Since 'original_train_takes' is a list of takes, splitting this list
# ensures no data leakage (clips from the same take stay together).
random.seed(42)  # Fix seed for reproducibility
random.shuffle(original_train_takes)

split_idx = int(len(original_train_takes) * 0.9)
new_train_takes = original_train_takes[:split_idx]
new_val_takes = original_train_takes[split_idx:]

for take in new_train_takes:
    take['split'] = 'train'

for take in new_val_takes:
    take['split'] = 'val'

# Combine all for processing
takes = new_train_takes + new_val_takes + original_val_takes

print(f"Data Split Summary:")
print(f"  Training:   {len(new_train_takes)} takes")
print(f"  Validation: {len(new_val_takes)} takes")
print(f"  Testing:    {len(original_val_takes)} takes")

label_map = {'good_executions': 0, 'tips_for_improvement': 1}

database = dict()
skipped_takes = []
seq_lens = []
counter_no_text = 0
counter_text = 0

# parse video annotations
for take in takes:
    take_uid = take['take_uid']
    print('Processing video {:s} ...'.format(take_uid))
    subset = take['split']
    if subset == 'train':
        subset = 'training'
    elif subset == 'val':
        subset = 'validation'
    elif subset == 'test':
        subset = 'testing'  # Map 'test' to 'testing'

    if take_uid not in takes_metadata:
        skipped_takes.append(take_uid)
        continue
    take_metadata = takes_metadata[take_uid]

    # get video annotations
    duration = take_metadata['duration_sec']  # clip length in second
    original_duration = take_metadata['duration_sec']
    frames = take_metadata['timesync_end_idx'] - take_metadata['timesync_start_idx']  # clip length in frame
    fps = frames / duration
    if fps < 10 or fps > 100:
        continue

    append_frames = append_sec = 0
    if (frames - clip_size) % stride:
        append_frames = stride - (frames - clip_size) % stride
        append_sec = append_frames / fps
        duration += append_sec
        frames += append_frames

    # save clip features
    si = 0
    ei = (frames - clip_size) // stride + 1

    for (video_type, video_file) in take['video_paths'].items():
        cid = take_uid + f'_{video_type}'
        # get video feature paths
        if video_type == 'ego':
            cam_name = video_file.split('/')[-1].split('.')[0][:4]
            omnivore_path = glob.glob(os.path.join(omnivore_dir, f"{take_uid}_{cam_name}*"))
        elif video_type[:3] == 'exo':
            cam_name = video_file.split('/')[-1].split('.')[0]
            omnivore_path = glob.glob(os.path.join(omnivore_dir, f"{take_uid}_{cam_name}_*"))
        else:
            raise ValueError

        # load video features
        if len(omnivore_path) != 1:  # or len(maws_path) != 1
            if len(omnivore_path) == 4:
                omnivore_path = [p for p in omnivore_path if '_rgb.pt' in p]
            else:
                raise ValueError(f">> Could not find unique feature file for take {take_uid} and video type {video_type}")
        omnivore_path = omnivore_path[0]

        if not os.path.exists(omnivore_path):
            print('> omnivore feature missing')
            continue
        omnivore_video = torch.load(omnivore_path).numpy()

        seq_lens.append(len(omnivore_video))

        if ei > len(omnivore_video):
            raise ValueError('end index exceeds omnivore feature length')

        # maws_clip = maws_video[si:ei]
        omnivore_clip = omnivore_video[si:ei]

        np.save(
            os.path.join(omnivore_out_dir, cid + '.npy'),
            omnivore_clip.astype(np.float32),
        )

        annotations = []

        # parse annotations from different annotators
        # annotators = clip['annotations']
        for class_type in ['good_executions', 'tips_for_improvement']:
            # parse action items
            items = take[class_type]
            for item in items:
                tpi = item['video_time']

                raw_list = item['list']
                if isinstance(raw_list, str):
                    try:
                        import ast
                        text_candidates = ast.literal_eval(raw_list)
                    except:
                        text_candidates = [raw_list]
                else:
                    text_candidates = raw_list

                text_list = []  # list of (cluster_id, text_n)
                if not isinstance(text_candidates, list):
                    text_candidates = [text_candidates]

                for text_c in text_candidates:
                    text_n = normalize_text(text_c)
                    if len(text_n) < 10:
                        continue
                    text_list.append(text_n)

                if len(text_list) == 0:
                    print(f'No valid texts for take {take_uid}, time {tpi}')
                    counter_no_text += 1
                    continue

                counter_text += 1
                annotations.append({
                    'label': class_type,
                    'reg_point': tpi,
                    'reg_point(frames)': int(tpi * fps),
                    'label_id': label_map[class_type],
                    'text_list': text_list,
                })

        if len(annotations) == 0:
            continue

        # build correct video file path and check if video exists
        video_file = video_file.split('/')
        video_file.insert(-1, 'downscaled/448')
        video_file = '/'.join(video_file)
        if not os.path.exists(f'{base_dir}/' + video_file):
            raise ValueError(f'Video file does not exist: {video_file}')

        egoexo_type = 'ego' if video_type == 'ego' else 'exo'
        database[cid] = {
            'video_file': video_file,
            'subset': subset,
            'egoexo_type': egoexo_type,
            'duration': round(duration, 2),
            'original_duration': round(original_duration, 2),
            'fps': round(fps, 2),
            'annotations': annotations,
            'good_executions': take['good_executions'],
            'tips_for_improvement': take['tips_for_improvement'],
            'task_name': take['task_name'],
            'scenario_name': take['scenario_name'],
        }

print(f'Counted {counter_text} annotations with text, {counter_no_text} without text.')
print(f'Skipped {len(skipped_takes)} takes due to missing metadata: {skipped_takes}')
print(f'Max sequence length: {max(seq_lens)}, min sequence length: {min(seq_lens)}, '
      f'average sequence length: {sum(seq_lens)/len(seq_lens):.1f}, std: {np.std(seq_lens):.1f}')
print(f'Sequence lengths: {seq_lens}')

out = {'version': 'v1', 'database': database}
with open(annot_out_path, 'w') as f:
    json.dump(out, f)