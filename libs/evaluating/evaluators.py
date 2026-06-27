import os
import json
import numpy as np
import pandas as pd
import ast

from collections import defaultdict

evaluators = {}
def register_evaluator(names):
    def decorator(cls):
        for name in names:
            if isinstance(name, (list, tuple)):
                for n in name:
                    evaluators[n] = cls
            else:
                evaluators[name] = cls
        return cls

    return decorator


def make_evaluator(name, ant_file, split, do_text_eval, **kwargs):
   return evaluators[name](ant_file, split, do_text_eval, **kwargs)


class BaseEvaluator(object):
    """
    Base class that handles loading GT from JSON files and printing results uniformly.
    """

    def __init__(
            self,
            ant_file,
            split,
            do_text_eval,
            egoexo_type,
            radius_thresholds=np.array([0.25, 0.50, 1.0]),
            top_k_recall=(1,),
            top_k_eval=None,
            num_workers=8,
            dataset_name=None,
            concat_views=False,
            feat_folder=None,
            file_prefix="",
            file_ext=".npy",
            **kwargs
    ):
        self.ant_file = ant_file
        self.split = split
        self.egoexo_type = egoexo_type
        self.concat_views = bool(concat_views)

        # Feature folder info for concat_views GT filtering
        if feat_folder is not None and not isinstance(feat_folder, (list, tuple)):
            feat_folder = (feat_folder,)
        self._feat_folder = feat_folder
        self._file_prefix = file_prefix or ""
        self._file_ext = file_ext

        self.ground_truth, self.video_to_scenario = self.load_ground_truth(ant_file, egoexo_type, split)
        self.radius_thresholds = np.array(radius_thresholds)
        if isinstance(top_k_recall, list):
            self.top_k_recall = tuple(top_k_recall)
        else:
            self.top_k_recall = top_k_recall
        self.top_k_eval = top_k_eval
        self.num_workers = num_workers
        if dataset_name is not None:
            self.dataset_name = dataset_name
        else:
            self.dataset_name = os.path.basename(ant_file).replace('.json', '')
        self.do_text_eval = do_text_eval

    @staticmethod
    def _take_uid_from_key(key):
        """Extract take_uid from database key (same logic as dataloader)."""
        for suffix in ('_ego', '_exo1', '_exo2', '_exo3', '_exo4'):
            if key.endswith(suffix):
                return key[:-len(suffix)]
        return key

    def load_ground_truth(self, json_file, egoexo_type, split):
        with open(json_file, "r", encoding="utf8") as f:
            data = json.load(f)
            json_db = data.get('database', data)

        if self.concat_views and egoexo_type in ('exo', 'both'):
            return self._load_ground_truth_concat(json_db, egoexo_type, split)

        vids, times, labels, texts = [], [], [], []
        video_to_scenario = {}

        # Hardcoded mapping based on your EgoExo4D logic
        label_map = {'good_executions': 0, 'tips_for_improvement': 1}

        for k, v in json_db.items():
            if (split is not None) and v['subset'].lower() != split: continue
            if egoexo_type == 'ego' and v['egoexo_type'] != 'ego': continue
            if egoexo_type == 'exo' and v['egoexo_type'] != 'exo': continue

            # Store scenario mapping
            if 'scenario_name' in v:
                video_to_scenario[k] = v['scenario_name']

            for key, label_id in label_map.items():
                for event in v.get(key, []):
                    # Time
                    times.append(float(event['video_time']))
                    # Text
                    raw_text = event.get('list', [])
                    if isinstance(raw_text, str):
                        # Handle stringified list "['text']"
                        try:
                            raw_text = ast.literal_eval(raw_text)
                        except:
                            raw_text = [raw_text]
                    if not isinstance(raw_text, list): raw_text = [str(raw_text)]

                    texts.append(raw_text)  # Keep as list for Multi-Ref BLEU
                    vids.append(k)
                    labels.append(label_id)

        gt_df = pd.DataFrame({
            'video-id': vids,
            'time': times,
            'label': labels,
            'text': texts
        })
        return gt_df, video_to_scenario

    def _check_sibling_feats_exist(self, sibling_ids):
        """
        Check that feature files exist for ALL sibling view IDs.
        Mirrors the same check the dataloader performs.
        Returns True only if all files exist.
        """
        if self._feat_folder is None:
            return True  # no feat info available, skip check
        for sid in sibling_ids:
            for folder in self._feat_folder:
                path = os.path.join(folder, self._file_prefix + sid + self._file_ext)
                if not os.path.exists(path):
                    return False
        return True

    def _load_ground_truth_concat(self, json_db, egoexo_type, split):
        """
        Load GT aligned to concat_views mode: one GT set per take_uid,
        keyed by the same reference view the dataloader uses.

        All views of the same take share identical annotations, so we
        pick ONE reference key per take and emit GT only under that key.
        This ensures prediction video-ids match GT video-ids.

        Applies the same filters as the dataloader:
          1. All expected view keys must exist in the database
          2. All sibling feature files must exist on disk

        Reference key logic (must match dataloader._load_json_db_concat):
          - egoexo_type='exo':  reference = {take_uid}_exo1
          - egoexo_type='both': reference = {take_uid}_ego
        """
        # Step 1: Group database entries by take_uid
        take_views = defaultdict(dict)  # take_uid -> {view_name: (key, value)}
        for k, v in json_db.items():
            if (split is not None) and v['subset'].lower() != split:
                continue
            take_uid = self._take_uid_from_key(k)
            if k.endswith('_ego'):
                view_name = 'ego'
            else:
                view_name = k[len(take_uid) + 1:]  # e.g., 'exo1'
            take_views[take_uid][view_name] = (k, v)

        # Step 2: Define expected views and reference view
        if egoexo_type == 'exo':
            expected_views = ['exo1', 'exo2', 'exo3', 'exo4']
            ref_view = 'exo1'
        else:  # 'both'
            expected_views = ['ego', 'exo1', 'exo2', 'exo3', 'exo4']
            ref_view = 'ego'

        # Step 3: Build GT using only the reference view per take
        vids, times, labels, texts = [], [], [], []
        video_to_scenario = {}
        label_map = {'good_executions': 0, 'tips_for_improvement': 1}
        n_skipped_views = 0
        n_skipped_feats = 0

        for take_uid, views in take_views.items():
            # Filter 1: Skip takes missing any expected view (same as dataloader)
            if not all(vn in views for vn in expected_views):
                n_skipped_views += 1
                continue

            # Filter 2: Skip takes with missing feature files (same as dataloader)
            sibling_ids = [views[v][0] for v in expected_views]
            if not self._check_sibling_feats_exist(sibling_ids):
                n_skipped_feats += 1
                continue

            ref_key, ref_value = views[ref_view]

            # Store scenario mapping
            if 'scenario_name' in ref_value:
                video_to_scenario[ref_key] = ref_value['scenario_name']

            for gt_key, label_id in label_map.items():
                for event in ref_value.get(gt_key, []):
                    times.append(float(event['video_time']))
                    raw_text = event.get('list', [])
                    if isinstance(raw_text, str):
                        try:
                            raw_text = ast.literal_eval(raw_text)
                        except:
                            raw_text = [raw_text]
                    if not isinstance(raw_text, list):
                        raw_text = [str(raw_text)]

                    texts.append(raw_text)
                    vids.append(ref_key)
                    labels.append(label_id)

        if n_skipped_views > 0 or n_skipped_feats > 0:
            print(f"[Evaluator concat_views] GT: {len(set(vids))} takes loaded, "
                  f"{n_skipped_views} skipped (missing views), "
                  f"{n_skipped_feats} skipped (missing features)")

        gt_df = pd.DataFrame({
            'video-id': vids,
            'time': times,
            'label': labels,
            'text': texts
        })
        return gt_df, video_to_scenario

    def _print_detection_metrics(self, output_str, mAP, average_mAP, mRecall, class_res=None):
        """
        Prints detection metrics.
        Requires class_res (from compute_classification_metrics) to show Match counts.
        """
        width_print = 35 + 10 * len(self.top_k_recall)

        # Build dynamic Recall Header (e.g., "R@1 | R@5")
        rec_headers = [f"R@{k:<2}" for k in self.top_k_recall]
        rec_header_str = " | ".join([f"{h:<7}" for h in rec_headers])

        output_str += "=" * width_print + "\n"
        output_str += "A. DETECTION METRICS (Finding the Action)\n"
        output_str += f"{'Radius':<8} | {'mAP':<6} | {'Match (t)':<10} | {rec_header_str}\n"
        output_str += "-" * width_print + "\n"

        avg_match_count = []
        for ridx, (rad, rad_mAP, rad_mRecall) in enumerate(zip(self.radius_thresholds, mAP, mRecall)):
            # 1. Get Match string if available
            match_str = "-"
            if class_res and rad in class_res:
                cnt = class_res[rad]['matched_count']
                total = len(self.ground_truth)
                match_str = f"{cnt}/{total}"
                avg_match_count.append(cnt)

            # 2. Start Row (Radius | mAP | Match)
            output_str += (
                f"{rad:<8.2f} | "
                f"{rad_mAP * 100:<6.2f} | "
                f"{match_str:<10}"
            )

            # 3. Add Recall columns
            for idx, k in enumerate(self.top_k_recall):
                output_str += f" | {rad_mRecall[idx] * 100:<7.2f}"
            output_str += "\n"

        output_str += "-" * width_print + "\n"

        # --- Average Row ---
        # Calculate Avg Match
        if avg_match_count:
            avg_cnt = int(np.mean(avg_match_count))
            avg_match_str = f"{avg_cnt}/{len(self.ground_truth)}"
        else:
            avg_match_str = "-"

        # Use double quotes for f-string to allow single quotes inside
        output_str += f"{'Average':<8} | {average_mAP * 100:<6.2f} | {avg_match_str:<10}"

        for idx, k in enumerate(self.top_k_recall):
            output_str += f" | {mRecall[:, idx].mean() * 100:<7.2f}"

        output_str += "\n" + "=" * width_print + "\n\n"

        return output_str

    def _print_classification_metrics(self, output_str, class_res):
        width_print = 70  # Adjusted to fit headers
        output_str += "=" * width_print + "\n"
        output_str += "B. CLASSIFICATION METRICS (Grading the Action)\n"
        output_str += f"{'Radius':<8} | {'Acc':<6} | {'BalAcc':<7} | {'MacF1':<6} | {'Prec(G)':<7} | {'Prec(B)':<7} | {'Match (t)':<9}\n"
        output_str += "-" * width_print + "\n"

        metrics_all = defaultdict(list)
        keys_order = ['accuracy', 'balanced_acc', 'macro_f1', 'precision_good', 'precision_bad', 'matched_count']
        for rad in self.radius_thresholds:
            if rad in class_res:
                m = class_res[rad]
                total_gt = len(self.ground_truth)

                output_str += (
                    f"{rad:<8.2f} | "
                    f"{m['accuracy']:<6.2f} | {m['balanced_acc']:<7.2f} | {m['macro_f1']:<6.2f} | "
                    f"{m['precision_good']:<7.2f} | {m['precision_bad']:<7.2f} | "
                    f"{m['matched_count']}/{total_gt:<5}\n"
                )

                # Accumulate only the keys we care about printing
                for key in keys_order:
                    metrics_all[key].append(m[key])

        output_str += "-" * width_print + "\n"
        output_str += f"{'Average':<8}"
        for key in keys_order:
            avg_val = np.mean(metrics_all[key])
            if 'count' in key:
                output_str += f" | {int(avg_val)}/{len(self.ground_truth):<5}"
            elif 'balanced' in key or 'precision' in key:
                output_str += f" | {avg_val:<7.2f}"
            else:
                output_str += f" | {avg_val:<6.2f}"

        output_str += "\n"
        output_str += "=" * width_print + "\n\n"

        return output_str

    def _print_text_generation_metrics(self, output_str, vlm_results):
        width_print = 75
        output_str += "=" * width_print + "\n"
        output_str += "C. CRITIQUE GENERATION (Text Quality)\n"

        # Header
        # Adjusted padding to align perfectly with data columns
        output_str += f"{'Radius':<8} | {'B-1':<6} | {'B-4':<6} | {'MET':<6} | {'R-L':<6} | {'BERT':<6}\n"
        output_str += "-" * width_print + "\n"

        # Accumulator for averages
        from collections import defaultdict
        metrics_all = defaultdict(list)

        # Exact order of keys to match the header columns
        keys_order = ['bleu1', 'bleu4', 'meteor', 'rougeL', 'bert_score']

        for radius in self.radius_thresholds:
            m = vlm_results[radius]

            # Print Row
            output_str += (
                f"{radius:<8.2f} | "
                f"{m['bleu1']:<6.2f} | {m['bleu4']:<6.2f} | "
                f"{m['meteor']:<6.2f} | {m['rougeL']:<6.2f} | {m['bert_score']:<6.2f}\n"
            )

            # Collect values for average
            for key in keys_order:
                metrics_all[key].append(m[key])

        # --- Average Row ---
        output_str += "-" * width_print + "\n"
        output_str += f"{'Average':<8}"

        for key in keys_order:
            avg_val = np.mean(metrics_all[key])
            output_str += f" | {avg_val:<6.2f}"

        output_str += "\n"
        output_str += "=" * width_print + "\n"

        return output_str