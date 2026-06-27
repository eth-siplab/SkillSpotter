import numpy as np
import pandas as pd
import torch
import logging
from joblib import Parallel, delayed

# NLP imports are only used for text-generation evaluation (the VLM
# --generate_text path). SkillSpotter does not generate text, so these are
# optional and imported lazily to avoid pulling in nltk/rouge/bert-score.
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer
    from bert_score import score as bert_score_func
except ImportError:
    sentence_bleu = SmoothingFunction = meteor_score = None
    rouge_scorer = bert_score_func = None


# -------------------------------------------------------------
# 1. mAP / Recall / DETECTION METRICS
# -------------------------------------------------------------
def interpolated_prec_rec(prec, rec):
    """Interpolated AP - VOCdevkit from VOC 2011."""
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap


def compute_average_precision_timestamp(ground_truth, prediction, radius_thresholds):
    """Compute AP for a single class (Thread-safe function)"""
    ap = np.zeros(len(radius_thresholds))
    if prediction.empty:
        return ap

    npos = float(len(ground_truth))
    lock_gt = [np.ones(len(ground_truth)) * -1 for _ in radius_thresholds]

    # Sort predictions by score
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.iloc[sort_idx].reset_index(drop=True)

    # Group GT by video for fast lookup
    ground_truth_gbvn = ground_truth.groupby('video-id')

    tp = np.zeros((len(radius_thresholds), len(prediction)))
    fp = np.zeros((len(radius_thresholds), len(prediction)))

    for idx, this_pred in prediction.iterrows():
        try:
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred['video-id'])
        except Exception:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        dists = np.abs(this_gt['time'].values - float(this_pred['time']))
        order = dists.argsort()

        for ridx, r in enumerate(radius_thresholds):
            matched = False
            for j in order:
                if dists[j] > r: break
                if lock_gt[ridx][this_gt.loc[j]['index']] >= 0: continue

                tp[ridx, idx] = 1
                lock_gt[ridx][this_gt.loc[j]['index']] = idx
                matched = True
                break
            if not matched:
                fp[ridx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(float)
    recall_cumsum = tp_cumsum / npos
    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for ridx in range(len(radius_thresholds)):
        ap[ridx] = interpolated_prec_rec(precision_cumsum[ridx, :], recall_cumsum[ridx, :])

    return ap


def wrapper_compute_mAP(preds, ground_truth, radius_thresholds, num_workers=8):
    """
    Wrapper to calculate mAP in parallel.
    Returns: mAP (array per radius), mAP_avg (float), mRecall (array)
    """
    # 1. Setup Activity Index (Classes)
    # We find all unique labels in GT to iterate over
    unique_labels = sorted(ground_truth['label'].unique())
    activity_index = {label: i for i, label in enumerate(unique_labels)}

    ap_matrix = np.zeros((len(radius_thresholds), len(activity_index)))

    # 2. Group by Label
    gt_by_label = ground_truth.groupby('label')
    pred_by_label = preds.groupby('label')

    # 3. Helper to get preds for a class safely
    def get_class_preds(lbl):
        try:
            return pred_by_label.get_group(lbl).reset_index(drop=True)
        except:
            return pd.DataFrame()

    # 4. Parallel Execution
    results = Parallel(n_jobs=num_workers)(
        delayed(compute_average_precision_timestamp)(
            ground_truth=gt_by_label.get_group(cls_id).reset_index(drop=True),
            prediction=get_class_preds(cls_id),
            radius_thresholds=radius_thresholds
        ) for cls_id in activity_index.keys()
    )

    for i, cls_id in enumerate(activity_index.keys()):
        ap_matrix[:, i] = results[i]

    # Calculate Summaries
    mAP = ap_matrix.mean(axis=1)
    average_mAP = mAP.mean()

    return mAP, average_mAP


def compute_topkx_recall_timestamp(
        ground_truth,
        prediction,
        radius_thresholds,
        top_k  # Tuple, e.g., (1, 5)
):
    """
    Recall@k×: for each class and video, take the top (k × #GT_in_video) predictions.
    Check if GT is recalled within radius.
    """
    if prediction.empty:
        return np.zeros((len(radius_thresholds), len(top_k)))

    tp = np.zeros((len(radius_thresholds), len(top_k)))
    n_gts = 0

    ground_truth_gbvn = ground_truth.groupby('video-id')
    prediction_gbvn = prediction.groupby('video-id')

    for videoid, _ in ground_truth_gbvn.groups.items():
        ground_truth_videoid = ground_truth_gbvn.get_group(videoid)
        n_gts += len(ground_truth_videoid)

        try:
            prediction_videoid = prediction_gbvn.get_group(videoid)
        except Exception:
            continue

        this_gt = ground_truth_videoid.reset_index()
        this_pred = prediction_videoid.reset_index()

        # Sort predictions by score
        score_sort_idx = this_pred['score'].values.argsort()[::-1]
        this_pred = this_pred.iloc[score_sort_idx].reset_index(drop=True)

        gt_times = this_gt['time'].values.astype(float)
        pred_times = this_pred['time'].values.astype(float)

        # Pairwise distance matrix [N_pred, N_gt]
        dmat = np.abs(pred_times[:, None] - gt_times[None, :])

        for ridx, r in enumerate(radius_thresholds):
            for kidx, k in enumerate(top_k):
                # We select top k * number_of_gt predictions
                # e.g., if video has 2 GTs and k=5, we check top 10 preds
                P = min(k * len(this_gt), len(this_pred))

                if P == 0: continue

                # Check if any of the top P preds hit the GTs
                # dmat[:P, :] shape is [P, N_gt]
                # (dmat <= r) is boolean match
                # .any(axis=0) means "Did this GT get hit by ANY valid pred?"
                ok = (dmat[:P, :] <= r)
                tp[ridx, kidx] += (ok.any(axis=0)).sum()

    recall = tp / max(n_gts, 1)
    return recall


def wrapper_compute_recall(preds, ground_truth, radius_thresholds, top_k, num_workers=8):
    """
    Wrapper to calculate Recall in parallel.
    """
    unique_labels = sorted(ground_truth['label'].unique())
    activity_index = {label: i for i, label in enumerate(unique_labels)}

    # Dimensions: [Radii, Top_K_Multipliers, Classes]
    recall_matrix = np.zeros((len(radius_thresholds), len(top_k), len(activity_index)))

    gt_by_label = ground_truth.groupby('label')
    pred_by_label = preds.groupby('label')

    def get_class_preds(lbl):
        try:
            return pred_by_label.get_group(lbl).reset_index(drop=True)
        except:
            return pd.DataFrame()

    results = Parallel(n_jobs=num_workers)(
        delayed(compute_topkx_recall_timestamp)(
            ground_truth=gt_by_label.get_group(cls_id).reset_index(drop=True),
            prediction=get_class_preds(cls_id),
            radius_thresholds=radius_thresholds,
            top_k=top_k
        ) for cls_id in activity_index.keys()
    )

    for i, cls_id in enumerate(activity_index.keys()):
        # result[i] is shape [Radii, TopK]
        recall_matrix[:, :, i] = results[i]

    # Average over classes (axis 2) -> Returns [Radii, TopK]
    mRecall = recall_matrix.mean(axis=2)
    return mRecall

# -------------------------------------------------------------
# 2. CLASSIFICATION METRICS (Global Greedy Matching)
# -------------------------------------------------------------
def compute_classification_metrics(preds, ground_truth, radius):
    """
    Matches predictions to GT using Greedy Bipartite Matching based on score.
    Calculates Accuracy, Balanced Acc, and F1.
    """
    # Prepare Data
    df_pred = pd.DataFrame(preds)

    # Filter Background predictions (Label 2)
    if 'label' in df_pred.columns:
        df_pred = df_pred[df_pred['label'] != 2].reset_index(drop=True)

    # Sort by Score (High -> Low) for Greedy Matching
    df_pred = df_pred.sort_values(by='score', ascending=False)

    gt_by_video = ground_truth.groupby('video-id')
    used_gt = set()

    stats = {
        'correct': 0, 'matched_count': 0,
        'tp_good': 0, 'fp_good': 0,
        'tp_bad': 0, 'fp_bad': 0
    }

    matched_pairs = []  # Store indices for text eval later

    for idx, pred in df_pred.iterrows():
        vid = pred['video-id']
        if vid not in gt_by_video.groups: continue

        this_gt = gt_by_video.get_group(vid)
        dists = np.abs(this_gt['time'].values - pred['time'])
        sorted_dist_indices = np.argsort(dists)

        matched_gt_idx = None

        # Find closest AVAILABLE GT
        for idx in sorted_dist_indices:
            if dists[idx] > radius: break
            global_gt_idx = this_gt.index[idx]

            if (vid, global_gt_idx) not in used_gt:
                matched_gt_idx = global_gt_idx
                used_gt.add((vid, global_gt_idx))
                break

        if matched_gt_idx is not None:
            # --- MATCH FOUND ---
            gt_row = ground_truth.loc[matched_gt_idx]
            stats['matched_count'] += 1

            # Save pair for text eval (if needed)
            matched_pairs.append((pred, gt_row))

            p_label = int(pred['label'])
            gt_label = int(gt_row['label'])

            if p_label == gt_label:
                stats['correct'] += 1
                if p_label == 0:
                    stats['tp_good'] += 1
                else:
                    stats['tp_bad'] += 1
            else:
                if p_label == 0:
                    stats['fp_good'] += 1
                else:
                    stats['fp_bad'] += 1

    # --- CALCULATION ---
    tp_g = stats['tp_good']
    fp_g = stats['fp_good']
    tp_b = stats['tp_bad']
    fp_b = stats['fp_bad']

    # Denominators for Recall (Total GTs)
    total_gt_good = tp_g + fp_b
    total_gt_bad = tp_b + fp_g

    total = stats['matched_count']

    # 1. Accuracy
    acc = (stats['correct'] / total * 100) if total > 0 else 0.0

    # 2. Balanced Accuracy
    rec_g = (tp_g / total_gt_good * 100) if total_gt_good > 0 else 0.0
    rec_b = (tp_b / total_gt_bad * 100) if total_gt_bad > 0 else 0.0
    bal_acc = (rec_g + rec_b) / 2

    # 3. F1 Scores
    prec_g = (tp_g / (tp_g + fp_g) * 100) if (tp_g + fp_g) > 0 else 0.0
    f1_g = (2 * prec_g * rec_g / (prec_g + rec_g)) if (prec_g + rec_g) > 0 else 0.0

    prec_b = (tp_b / (tp_b + fp_b) * 100) if (tp_b + fp_b) > 0 else 0.0
    f1_b = (2 * prec_b * rec_b / (prec_b + rec_b)) if (prec_b + rec_b) > 0 else 0.0

    macro_f1 = (f1_g + f1_b) / 2

    results = {
        'accuracy': acc,
        'balanced_acc': bal_acc,
        'macro_f1': macro_f1,
        'precision_good': prec_g,
        'precision_bad': prec_b,
        'f1_good': f1_g,
        'f1_bad': f1_b,
        'matched_count': stats['matched_count'],
        'total_gt': len(ground_truth),
        'matched_pairs': matched_pairs  # Return actual rows for text eval
    }
    return results


# -------------------------------------------------------------
# 3. TEXT METRICS
# -------------------------------------------------------------
def compute_text_metrics(matched_pairs):
    """
    Calculates NLP metrics on the matched pairs identified by the classification step.
    """
    if not matched_pairs:
        return {'bleu1': 0, 'bleu4': 0, 'meteor': 0, 'rougeL': 0, 'bert_score': 0}

    smoother = SmoothingFunction().method1
    scorer_rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    stats = {'bleu1': [], 'bleu4': [], 'meteor': [], 'rouge': []}
    bert_cands, bert_refs = [], []

    for pred_row, gt_row in matched_pairs:
        p_text = str(pred_row.get('pred_text', "")).strip().lower()

        # GT Handling (List or String)
        gt_list = gt_row['text']
        if isinstance(gt_list, str): gt_list = [gt_list]
        gt_list = [t for t in gt_list if len(t.strip()) > 0]

        if len(p_text) > 0 and len(gt_list) > 0:
            hyp_tokens = p_text.split()
            ref_tokens_list = [t.strip().lower().split() for t in gt_list]

            try:
                # BLEU
                stats['bleu1'].append(
                    sentence_bleu(ref_tokens_list, hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smoother))
                stats['bleu4'].append(sentence_bleu(ref_tokens_list, hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25),
                                                    smoothing_function=smoother))
                # METEOR
                stats['meteor'].append(meteor_score(ref_tokens_list, hyp_tokens))
            except:
                pass

            # ROUGE (Max over refs)
            r_scores = [scorer_rouge.score(r, p_text)['rougeL'].fmeasure for r in gt_list]
            stats['rouge'].append(max(r_scores))

            # BERT
            bert_cands.append(p_text)
            bert_refs.append(gt_list)

    # BERTScore Batch
    bert_val = 0.0
    if len(bert_cands) > 0:
        try:
            if torch.cuda.is_available():
                device = f"cuda:{torch.cuda.current_device()}"
            else:
                device = "cpu"
                print("BERTScore is being calculated on CPU, which may be slow.")

            _, _, F1 = bert_score_func(bert_cands, bert_refs, lang="en", verbose=False, device=device, batch_size=64)
            bert_val = F1.mean().item() * 100
        except Exception as e:
            print(f"Warning: BERTScore failed: {e}")

    return {
        'bleu1': np.mean(stats['bleu1']) * 100 if stats['bleu1'] else 0.0,
        'bleu4': np.mean(stats['bleu4']) * 100 if stats['bleu4'] else 0.0,
        'meteor': np.mean(stats['meteor']) * 100 if stats['meteor'] else 0.0,
        'rougeL': np.mean(stats['rouge']) * 100 if stats['rouge'] else 0.0,
        'bert_score': bert_val
    }

# -------------------------------------------------------------
# 4. HELPER FUNCTIONS
# -------------------------------------------------------------
def slice_predictions(preds, k):
    """
    Keeps only the top K predictions per video.
    preds: DataFrame or Dict containing 'video-id' and 'score'.
    """
    if k is None or k <= 0:
        return preds

    # Ensure DataFrame
    if not isinstance(preds, pd.DataFrame):
        preds = pd.DataFrame(preds)

    print(f"[Evaluator] Slicing predictions to Top-{k} per video...")

    # Sort Descending
    preds = preds.sort_values(by=['video-id', 'score'], ascending=[True, False])

    # Slice
    return preds.groupby('video-id').head(k).reset_index(drop=True)


def plot_pr_curve(preds, ground_truth, title_add=None, output_path='pr_curve.png'):
    """
    Generates and saves the Precision-Recall curve comparing Good vs Bad classes.
    """
    import matplotlib.pyplot as plt

    # Ensure preds is sorted
    preds = preds.sort_values(by='score', ascending=False).reset_index(drop=True)

    plt.figure(figsize=(10, 6))

    # We only care about Good (0) and Bad (1)
    classes = {0: 'Good', 1: 'Bad'}
    colors = {0: 'blue', 1: 'red'}

    # We use the standard Radius (e.g., 1.0s or the last one in the list)
    # Default to 1.0s for plotting clarity
    plot_radius = 1.0
    for cls_id, cls_name in classes.items():
        # Filter Preds
        pred_cls = preds[preds['label'] == cls_id]

        # Filter GT (Get total count for Recall denom)
        gt_total = len(ground_truth[ground_truth['label'] == cls_id])

        if gt_total == 0 or len(pred_cls) == 0:
            continue

        # Match Logic (Vectorized-ish)
        tp = np.zeros(len(pred_cls))
        fp = np.zeros(len(pred_cls))

        # Optimization: Pre-group GT
        gt_by_vid = ground_truth[ground_truth['label'] == cls_id].groupby('video-id')
        gt_matched = set()

        for idx, (i, row) in enumerate(pred_cls.iterrows()):
            vid = row['video-id']
            if vid not in gt_by_vid.groups:
                fp[idx] = 1
                continue

            this_gt = gt_by_vid.get_group(vid)
            dists = np.abs(this_gt['time'].values - row['time'])

            if dists.min() <= plot_radius:
                # Check for double match
                match_idx = dists.argmin()
                global_id = (vid, this_gt.index[match_idx])

                if global_id not in gt_matched:
                    tp[idx] = 1
                    gt_matched.add(global_id)
                else:
                    fp[idx] = 1  # Duplicate detection
            else:
                fp[idx] = 1

        # Calculate Curves
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        recalls = tp_cumsum / gt_total

        # Plot
        plt.plot(recalls, precisions, label=f'{cls_name} (Total GT: {gt_total})', color=colors[cls_id], linewidth=2)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve {title_add} (Radius {plot_radius}s)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])

    plt.savefig(output_path)
    print(f"[Evaluator] PR Curve saved to {output_path}")
    plt.close()