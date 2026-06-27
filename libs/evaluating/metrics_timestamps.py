import numpy as np
import pandas as pd
from collections import defaultdict

from .evaluators import register_evaluator, BaseEvaluator
from .evaluation_utils import (
    wrapper_compute_mAP,
    wrapper_compute_recall,
    slice_predictions,
)


@register_evaluator(["skillspotter"])
class TimestampRegression(BaseEvaluator):
    """
    ActionFormer timestamp regression evaluator (binary Good/Bad labels).

    Adds:
      - mAP(Agns): class-agnostic timestamp finding (label=0 for all preds/gt)
      - mAP(Spec): class-specific Good/Bad mAP (uses original labels)
      - Rec(G), Rec(B) + TP/FP/TN/FN on matched pairs (GOOD=positive)
    """

    def __init__(self, ant_file, split, do_text_eval, **kwargs):
        super().__init__(ant_file, split, do_text_eval, **kwargs)

    def evaluate(self, preds, verbose=True, metrics_file=None):
        # 1) Format Predictions
        if isinstance(preds, dict):
            preds = pd.DataFrame(preds)

        print(f"[Evaluator] Number of predictions: {len(preds)}")

        # 2) Slice (Optional)
        if self.top_k_eval is not None:
            preds = slice_predictions(preds, self.top_k_eval)

        # ---- A) Detection metrics (two views) ----
        # A1) Class-agnostic (Finding): label=0 for everything
        preds_agn = preds.copy()
        gt_agn = self.ground_truth.copy()
        preds_agn["label"] = 0
        gt_agn["label"] = 0

        mAP_agn, avg_mAP_agn = wrapper_compute_mAP(
            preds_agn, gt_agn, self.radius_thresholds, self.num_workers
        )
        mRecall = wrapper_compute_recall(
            preds_agn, gt_agn, self.radius_thresholds, self.top_k_recall, num_workers=self.num_workers
        )

        # A2) Class-specific (Grading mAP): Good/Bad labels must match
        # Uses original labels in preds and ground_truth
        mAP_spec, avg_mAP_spec = wrapper_compute_mAP(
            preds, self.ground_truth, self.radius_thresholds, self.num_workers
        )

        # ---- B) Classification metrics (on matched pairs) ----
        class_res = {}
        for r in self.radius_thresholds:
            class_res[float(r)] = self._compute_classification_metrics_binary(preds, self.ground_truth, float(r))

        # ---- C) Per-scenario breakdown ----
        scenario_results = {}
        if hasattr(self, 'video_to_scenario') and self.video_to_scenario:
            scenario_results = self._compute_per_scenario_metrics(preds)

        # ---- Print ----
        if verbose:
            output_str = f"\n[RESULTS] ActionFormer Evaluation on {self.dataset_name}\n"
            output_str = self._print_dual_map_metrics(
                output_str,
                np.array(mAP_agn), float(avg_mAP_agn),
                np.array(mAP_spec), float(avg_mAP_spec),
                np.array(mRecall),
            )
            output_str = self._print_classification_metrics_binary(output_str, class_res)

            if scenario_results:
                output_str = self._print_per_scenario_metrics(output_str, scenario_results)

            print(output_str)

            if metrics_file:
                print(f"Saving metrics to {metrics_file}")
                with open(metrics_file, "w") as f:
                    f.write(output_str)

        # Return the original signature (and include extras in case your caller wants them)
        return {
            "mAP_agnostic": avg_mAP_agn,
            "mAP_specific": avg_mAP_spec,
            "mAP_per_radius_agnostic": np.array(mAP_agn),
            "mAP_per_radius_specific": np.array(mAP_spec),
            "mRecall": np.array(mRecall),
            "class_results": class_res,
            "scenario_results": scenario_results,
        }

    # ---------------------------------------------------------------------
    # Matching + binary classification metrics on matched pairs (GOOD=positive)
    # ---------------------------------------------------------------------
    def _compute_classification_metrics_binary(self, preds_df, gt_df, radius):
        """
        Greedy match per video (highest score first), within +/- radius seconds.

        Confusion treats GOOD as positive:
          TP: pred good, gt good
          FP: pred good, gt bad
          TN: pred bad,  gt bad
          FN: pred bad,  gt good
        """
        # Defensive: ensure required columns exist
        for col in ("video-id", "time", "label", "score"):
            if col not in preds_df.columns:
                raise KeyError(f"[TimestampRegression] preds missing column: {col}")
        for col in ("video-id", "time", "label"):
            if col not in gt_df.columns:
                raise KeyError(f"[TimestampRegression] ground_truth missing column: {col}")

        matches = self._match_predictions_to_gt(preds_df, gt_df, radius)

        TP = FP = TN = FN = 0

        for m in matches:
            pred_label = int(m["pred_label"])
            gt_label = int(m["gt_label"])

            # Expect binary {0,1}. If not, still treat: 0=good, else=bad.
            pred_q = 0 if pred_label == 0 else 1
            gt_q = 0 if gt_label == 0 else 1

            if gt_q == 0 and pred_q == 0:
                TP += 1
            elif gt_q == 1 and pred_q == 0:
                FP += 1
            elif gt_q == 1 and pred_q == 1:
                TN += 1
            elif gt_q == 0 and pred_q == 1:
                FN += 1

        used = len(matches)

        acc = self._safe_div((TP + TN), used) * 100.0

        rec_g = self._safe_div(TP, (TP + FN))          # TPR good
        rec_b = self._safe_div(TN, (TN + FP))          # TPR bad (specificity for good)
        bal_acc = 0.5 * (rec_g + rec_b) * 100.0

        prec_g = self._safe_div(TP, (TP + FP))
        prec_b = self._safe_div(TN, (TN + FN))         # precision when treating bad as positive

        f1_g = self._safe_div(2 * prec_g * rec_g, (prec_g + rec_g))
        f1_b = self._safe_div(2 * prec_b * rec_b, (prec_b + rec_b))
        mac_f1 = 0.5 * (f1_g + f1_b) * 100.0

        # Detection recall: fraction of GT events matched by any prediction
        total_gt = int(len(gt_df))
        det_recall = self._safe_div(used, total_gt) * 100.0

        # Detection precision: fraction of predictions that matched a GT event
        total_preds = int(len(preds_df))
        det_precision = self._safe_div(used, total_preds) * 100.0

        return {
            "acc": acc,
            "bal_acc": bal_acc,
            "mac_f1": mac_f1,
            "det_recall": det_recall,
            "det_precision": det_precision,
            "prec_g": prec_g * 100.0,
            "prec_b": prec_b * 100.0,
            "rec_g": rec_g * 100.0,
            "rec_b": rec_b * 100.0,
            "tp": TP,
            "fp": FP,
            "tn": TN,
            "fn": FN,
            "matched_count": used,
            "total_gt": total_gt,
            "total_preds": total_preds,
        }

    def _match_predictions_to_gt(self, preds_df, gt_df, radius):
        """
        Greedy matching (per video):
          - sort predictions by score desc
          - match each pred to closest unmatched GT within radius
        Returns list of matches (one per matched pred).
        """
        pred_by_video = defaultdict(list)
        for _, row in preds_df.iterrows():
            pred_by_video[row["video-id"]].append({
                "time": float(row["time"]),
                "score": float(row["score"]),
                "label": int(row["label"]),
            })

        gt_by_video = defaultdict(list)
        for _, row in gt_df.iterrows():
            gt_by_video[row["video-id"]].append({
                "time": float(row["time"]),
                "label": int(row["label"]),
            })

        matches = []
        for vid, preds in pred_by_video.items():
            preds = sorted(preds, key=lambda x: -x["score"])
            gts = gt_by_video.get(vid, [])
            gt_matched = [False] * len(gts)

            for pred in preds:
                best_gt_idx = -1
                best_dist = float("inf")

                for gt_idx, gt in enumerate(gts):
                    if gt_matched[gt_idx]:
                        continue
                    dist = abs(pred["time"] - gt["time"])
                    if dist <= radius and dist < best_dist:
                        best_dist = dist
                        best_gt_idx = gt_idx

                if best_gt_idx >= 0:
                    gt_matched[best_gt_idx] = True
                    matches.append({
                        "video_id": vid,
                        "pred_time": pred["time"],
                        "pred_label": pred["label"],
                        "pred_score": pred["score"],
                        "gt_time": gts[best_gt_idx]["time"],
                        "gt_label": gts[best_gt_idx]["label"],
                        "time_diff": best_dist,
                    })

        return matches

    # ---------------------------------------------------------------------
    # Printing (same table style you asked for)
    # ---------------------------------------------------------------------
    def _print_dual_map_metrics(self, output_str, map_ag, avg_ag, map_sp, avg_sp, mRecall):
        width = 50
        output_str += "=" * width + "\n"
        output_str += "A. DETECTION METRICS (Finding the Action)\n"
        output_str += f"{'Radius':<8} | {'mAP(Agns)':<10} | {'mAP(Spec)':<10} | {'R@1':<6}\n"
        output_str += "-" * width + "\n"

        for i, rad in enumerate(self.radius_thresholds):
            r1 = (mRecall[i, 0] * 100.0) if (mRecall.ndim > 1) else (mRecall[i] * 100.0)
            output_str += (
                f"{rad:<8.2f} | "
                f"{map_ag[i] * 100.0:<10.2f} | "
                f"{map_sp[i] * 100.0:<10.2f} | "
                f"{r1:<6.2f}\n"
            )

        output_str += "-" * width + "\n"
        r1_avg = (np.mean(mRecall[:, 0]) * 100.0) if (mRecall.ndim > 1) else (np.mean(mRecall) * 100.0)
        output_str += (
            f"{'Average':<8} | "
            f"{avg_ag * 100.0:<10.2f} | "
            f"{avg_sp * 100.0:<10.2f} | "
            f"{r1_avg:<6.2f}\n"
        )
        output_str += "=" * width + "\n\n"
        return output_str

    def _print_classification_metrics_binary(self, output_str, class_res):
        width = 125
        output_str += "=" * width + "\n"
        output_str += "B. CLASSIFICATION METRICS (Grading the Action)\n"
        output_str += (
            f"{'Radius':<8} | {'Acc':<6} | {'BalAcc':<7} | {'MacF1':<6} | {'Recall':<7} | {'Prec':<7} | "
            f"{'Prec(G)':<7} | {'Prec(B)':<7} | {'Rec(G)':<7} | {'Rec(B)':<7} | {'Match (t)'}\n"
        )
        output_str += "-" * width + "\n"

        accs, bals, f1s, recalls, precisions, pg, pb, rg, rb, ms = [], [], [], [], [], [], [], [], [], []
        tp_sum = fp_sum = tn_sum = fn_sum = 0

        for rad in self.radius_thresholds:
            r = class_res[float(rad)]
            match_str = f"{r['matched_count']}/{r['total_gt']}"

            output_str += (
                f"{rad:<8.2f} | "
                f"{r['acc']:<6.2f} | "
                f"{r['bal_acc']:<7.2f} | "
                f"{r['mac_f1']:<6.2f} | "
                f"{r['det_recall']:<7.2f} | "
                f"{r['det_precision']:<7.2f} | "
                f"{r['prec_g']:<7.2f} | "
                f"{r['prec_b']:<7.2f} | "
                f"{r['rec_g']:<7.2f} | "
                f"{r['rec_b']:<7.2f} | "
                f"{match_str}\n"
            )

            accs.append(r["acc"])
            bals.append(r["bal_acc"])
            f1s.append(r["mac_f1"])
            recalls.append(r["det_recall"])
            precisions.append(r["det_precision"])
            pg.append(r["prec_g"])
            pb.append(r["prec_b"])
            rg.append(r["rec_g"])
            rb.append(r["rec_b"])
            ms.append(r["matched_count"])

            tp_sum += r["tp"]
            fp_sum += r["fp"]
            tn_sum += r["tn"]
            fn_sum += r["fn"]

        output_str += "-" * width + "\n"
        total_gt = class_res[float(self.radius_thresholds[0])]["total_gt"]
        output_str += (
            f"{'Average':<8} | "
            f"{np.mean(accs):<6.2f} | "
            f"{np.mean(bals):<7.2f} | "
            f"{np.mean(f1s):<6.2f} | "
            f"{np.mean(recalls):<7.2f} | "
            f"{np.mean(precisions):<7.2f} | "
            f"{np.mean(pg):<7.2f} | "
            f"{np.mean(pb):<7.2f} | "
            f"{np.mean(rg):<7.2f} | "
            f"{np.mean(rb):<7.2f} | "
            f"{int(np.mean(ms))}/{total_gt}\n"
        )
        output_str += "=" * width + "\n"
        output_str += f"Confusion totals across radii (GOOD=positive): TP={tp_sum}, FP={fp_sum}, TN={tn_sum}, FN={fn_sum}\n\n"
        return output_str

    # ---------------------------------------------------------------------
    # Per-scenario evaluation
    # ---------------------------------------------------------------------
    def _compute_per_scenario_metrics(self, preds):
        """
        Compute mAP-Agn, mAP-Spec, BA, and F1 per scenario.
        Uses self.video_to_scenario to map video-ids to scenarios.
        """
        v2s = self.video_to_scenario
        gt = self.ground_truth

        # Build scenario column for GT
        gt_scenarios = gt["video-id"].map(v2s)
        # Build scenario column for preds
        pred_scenarios = preds["video-id"].map(v2s)

        scenarios = sorted(gt_scenarios.dropna().unique())
        if len(scenarios) == 0:
            return {}

        results = {}
        for scenario in scenarios:
            # Filter GT and preds to this scenario
            gt_mask = gt_scenarios == scenario
            pred_mask = pred_scenarios == scenario
            gt_scen = gt.loc[gt_mask].reset_index(drop=True)
            preds_scen = preds.loc[pred_mask].reset_index(drop=True)

            if len(gt_scen) == 0:
                continue

            # mAP-Agnostic
            gt_agn = gt_scen.copy()
            preds_agn = preds_scen.copy()
            gt_agn["label"] = 0
            preds_agn["label"] = 0
            try:
                _, avg_mAP_agn = wrapper_compute_mAP(
                    preds_agn, gt_agn, self.radius_thresholds, self.num_workers
                )
            except Exception:
                avg_mAP_agn = 0.0

            # mAP-Specific
            try:
                _, avg_mAP_spec = wrapper_compute_mAP(
                    preds_scen, gt_scen, self.radius_thresholds, self.num_workers
                )
            except Exception:
                avg_mAP_spec = 0.0

            # Classification (use middle radius for summary; compute for all and average)
            ba_vals, f1_vals, recall_vals, precision_vals = [], [], [], []
            for r in self.radius_thresholds:
                cls = self._compute_classification_metrics_binary(preds_scen, gt_scen, float(r))
                ba_vals.append(cls["bal_acc"])
                f1_vals.append(cls["mac_f1"])
                recall_vals.append(cls["det_recall"])
                precision_vals.append(cls["det_precision"])

            n_gt_good = int((gt_scen["label"] == 0).sum())
            n_gt_bad = int((gt_scen["label"] == 1).sum())

            results[scenario] = {
                "mAP_agn": avg_mAP_agn * 100.0,
                "mAP_spec": avg_mAP_spec * 100.0,
                "bal_acc": float(np.mean(ba_vals)),
                "mac_f1": float(np.mean(f1_vals)),
                "det_recall": float(np.mean(recall_vals)),
                "det_precision": float(np.mean(precision_vals)),
                "n_gt": len(gt_scen),
                "n_gt_good": n_gt_good,
                "n_gt_bad": n_gt_bad,
                "n_preds": len(preds_scen),
            }

        return results

    def _print_per_scenario_metrics(self, output_str, scenario_results):
        width = 120
        output_str += "=" * width + "\n"
        output_str += "C. PER-SCENARIO BREAKDOWN (Averaged over radius thresholds)\n"
        output_str += (
            f"{'Scenario':<16} | {'mAP(Agn)':<9} | {'mAP(Spec)':<10} | "
            f"{'BalAcc':<7} | {'MacF1':<6} | {'Recall':<7} | {'Prec':<7} | "
            f"{'#GT':<5} | {'#Good':<6} | {'#Bad':<5} | {'#Preds':<7}\n"
        )
        output_str += "-" * width + "\n"

        # Sort scenarios alphabetically
        for scenario in sorted(scenario_results.keys()):
            r = scenario_results[scenario]
            output_str += (
                f"{scenario:<16} | "
                f"{r['mAP_agn']:<9.2f} | "
                f"{r['mAP_spec']:<10.2f} | "
                f"{r['bal_acc']:<7.2f} | "
                f"{r['mac_f1']:<6.2f} | "
                f"{r['det_recall']:<7.2f} | "
                f"{r['det_precision']:<7.2f} | "
                f"{r['n_gt']:<5} | "
                f"{r['n_gt_good']:<6} | "
                f"{r['n_gt_bad']:<5} | "
                f"{r['n_preds']:<7}\n"
            )

        # Average row
        n = len(scenario_results)
        if n > 0:
            output_str += "-" * width + "\n"
            avg_agn = np.mean([r["mAP_agn"] for r in scenario_results.values()])
            avg_spec = np.mean([r["mAP_spec"] for r in scenario_results.values()])
            avg_ba = np.mean([r["bal_acc"] for r in scenario_results.values()])
            avg_f1 = np.mean([r["mac_f1"] for r in scenario_results.values()])
            avg_recall = np.mean([r["det_recall"] for r in scenario_results.values()])
            avg_precision = np.mean([r["det_precision"] for r in scenario_results.values()])
            tot_gt = sum(r["n_gt"] for r in scenario_results.values())
            tot_good = sum(r["n_gt_good"] for r in scenario_results.values())
            tot_bad = sum(r["n_gt_bad"] for r in scenario_results.values())
            tot_preds = sum(r["n_preds"] for r in scenario_results.values())
            output_str += (
                f"{'Macro-Avg':<16} | "
                f"{avg_agn:<9.2f} | "
                f"{avg_spec:<10.2f} | "
                f"{avg_ba:<7.2f} | "
                f"{avg_f1:<6.2f} | "
                f"{avg_recall:<7.2f} | "
                f"{avg_precision:<7.2f} | "
                f"{tot_gt:<5} | "
                f"{tot_good:<6} | "
                f"{tot_bad:<5} | "
                f"{tot_preds:<7}\n"
            )

        output_str += "=" * width + "\n\n"
        return output_str

    def _safe_div(self, a, b):
        return float(a) / float(b) if b else 0.0

