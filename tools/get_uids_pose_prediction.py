import os, json

with open('/path/to/EgoExo4D/annotations/egoexo4d.json') as f:
    data = json.load(f)

base_uids = set()
for rid in data['database'].keys():
    for suffix in ['_ego', '_exo1', '_exo2', '_exo3', '_exo4']:
        if rid.endswith(suffix):
            base_uids.add(rid[:-len(suffix)])
            break
    else:
        base_uids.add(rid)

cam_root = '/path/to/EgoExo4D/annotations/ego_pose'
have_cam = set()
for split in ['train', 'val']:
    d = os.path.join(cam_root, split, 'camera_pose')
    if os.path.isdir(d):
        for fn in os.listdir(d):
            if fn.endswith('.json'):
                have_cam.add(fn.replace('.json', ''))

needed = sorted(base_uids & have_cam)
out_path = '/path/to/EgoExo4D/needed_take_uids.txt'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w') as f:
    for uid in needed:
        f.write(uid + '\n')
print(f'Written {len(needed)} take_uids to {out_path}')
print(f'  ({len(base_uids)} in dataloader, {len(have_cam)} have camera_pose, {len(needed)} overlap)')
