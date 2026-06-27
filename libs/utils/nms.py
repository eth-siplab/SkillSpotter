# Functions for 1D NMS, modified from:
# https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/nms.py
import torch

# The compiled C++ segment-NMS op (nms_1d_cpu) is OPTIONAL. SkillSpotter runs
# with nms_method="none" and uses the pure-PyTorch timestamp NMS below, so the
# extension is not required to import or run the model. We only raise if a
# function that genuinely needs it is actually called.
try:
    import nms_1d_cpu
except ImportError:
    nms_1d_cpu = None

_NMS_BUILD_HINT = (
    "The 'nms_1d_cpu' C++ extension is not built. It is only needed for "
    "segment Soft/Hard-NMS (nms_method='soft'/'hard'). SkillSpotter uses "
    "nms_method='none', so you normally do not need it. To build it: "
    "cd libs/utils && python setup.py install"
)


class NMSop(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, segs, scores, cls_idxs,
        iou_threshold, min_score, max_num
    ):
        # vanilla nms will not change the score, so we can filter segs first
        is_filtering_by_score = (min_score > 0)
        if is_filtering_by_score:
            valid_mask = scores > min_score
            segs, scores = segs[valid_mask], scores[valid_mask]
            cls_idxs = cls_idxs[valid_mask]
            valid_inds = torch.nonzero(
                valid_mask, as_tuple=False).squeeze(dim=1)

        # nms op; return inds that is sorted by descending order
        if nms_1d_cpu is None:
            raise ImportError(_NMS_BUILD_HINT)
        inds = nms_1d_cpu.nms(
            segs.contiguous().cpu(),
            scores.contiguous().cpu(),
            iou_threshold=float(iou_threshold))
        # cap by max number
        if max_num > 0:
            inds = inds[:min(max_num, len(inds))]
        # return the sorted segs / scores
        sorted_segs = segs[inds]
        sorted_scores = scores[inds]
        sorted_cls_idxs = cls_idxs[inds]
        return sorted_segs.clone(), sorted_scores.clone(), sorted_cls_idxs.clone()


class SoftNMSop(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, segs, scores, cls_idxs,
        iou_threshold, sigma, min_score, method, max_num
    ):
        # pre allocate memory for sorted results
        dets = segs.new_empty((segs.size(0), 3), device='cpu')
        # softnms op, return dets that stores the sorted segs / scores
        if nms_1d_cpu is None:
            raise ImportError(_NMS_BUILD_HINT)
        inds = nms_1d_cpu.softnms(
            segs.cpu(),
            scores.cpu(),
            dets.cpu(),
            iou_threshold=float(iou_threshold),
            sigma=float(sigma),
            min_score=float(min_score),
            method=int(method))
        # cap by max number
        if max_num > 0:
            n_segs = min(len(inds), max_num)
        else:
            n_segs = len(inds)
        sorted_segs = dets[:n_segs, :2]
        sorted_scores = dets[:n_segs, 2]
        sorted_cls_idxs = cls_idxs[inds]
        sorted_cls_idxs = sorted_cls_idxs[:n_segs]
        return sorted_segs.clone(), sorted_scores.clone(), sorted_cls_idxs.clone()


def seg_voting(nms_segs, all_segs, all_scores, iou_threshold, score_offset=1.5):
    """
        blur localization results by incorporating side segs.
        this is known as bounding box voting in object detection literature.
        slightly boost the performance around iou_threshold
    """

    # *_segs : N_i x 2, all_scores: N,
    # apply offset
    offset_scores = all_scores + score_offset

    # computer overlap between nms and all segs
    # construct the distance matrix of # N_nms x # N_all
    num_nms_segs, num_all_segs = nms_segs.shape[0], all_segs.shape[0]
    ex_nms_segs = nms_segs[:, None].expand(num_nms_segs, num_all_segs, 2)
    ex_all_segs = all_segs[None, :].expand(num_nms_segs, num_all_segs, 2)

    # compute intersection
    left = torch.maximum(ex_nms_segs[:, :, 0], ex_all_segs[:, :, 0])
    right = torch.minimum(ex_nms_segs[:, :, 1], ex_all_segs[:, :, 1])
    inter = (right-left).clamp(min=0)

    # lens of all segments
    nms_seg_lens = ex_nms_segs[:, :, 1] - ex_nms_segs[:, :, 0]
    all_seg_lens = ex_all_segs[:, :, 1] - ex_all_segs[:, :, 0]

    # iou
    iou = inter / (nms_seg_lens + all_seg_lens - inter)

    # get neighbors (# N_nms x # N_all) / weights
    seg_weights = (iou >= iou_threshold).to(all_scores.dtype) * all_scores[None, :] * iou
    seg_weights /= torch.sum(seg_weights, dim=1, keepdim=True)
    refined_segs = seg_weights @ all_segs

    return refined_segs

def batched_nms(
    segs,
    scores,
    cls_idxs,
    iou_threshold,
    min_score,
    max_seg_num,
    use_soft_nms=True,
    multiclass=True,
    sigma=0.5,
    voting_thresh=0.75,
):
    # Based on Detectron2 implementation,
    num_segs = segs.shape[0]
    # corner case, no prediction outputs
    if num_segs == 0:
        return torch.zeros([0, 2]),\
               torch.zeros([0,]),\
               torch.zeros([0,], dtype=cls_idxs.dtype)

    if multiclass:
        # multiclass nms: apply nms on each class independently
        new_segs, new_scores, new_cls_idxs = [], [], []
        for class_id in torch.unique(cls_idxs):
            curr_indices = torch.where(cls_idxs == class_id)[0]
            # soft_nms vs nms
            if use_soft_nms:
                sorted_segs, sorted_scores, sorted_cls_idxs = SoftNMSop.apply(
                    segs[curr_indices],
                    scores[curr_indices],
                    cls_idxs[curr_indices],
                    iou_threshold,
                    sigma,
                    min_score,
                    2,
                    max_seg_num
                )
            else:
                sorted_segs, sorted_scores, sorted_cls_idxs = NMSop.apply(
                    segs[curr_indices],
                    scores[curr_indices],
                    cls_idxs[curr_indices],
                    iou_threshold,
                    min_score,
                    max_seg_num
                )
            # disable seg voting for multiclass nms, no sufficient segs

            # fill in the class index
            new_segs.append(sorted_segs)
            new_scores.append(sorted_scores)
            new_cls_idxs.append(sorted_cls_idxs)

        # cat the results
        new_segs = torch.cat(new_segs)
        new_scores = torch.cat(new_scores)
        new_cls_idxs = torch.cat(new_cls_idxs)

    else:
        # class agnostic
        if use_soft_nms:
            new_segs, new_scores, new_cls_idxs = SoftNMSop.apply(
                segs, scores, cls_idxs, iou_threshold,
                sigma, min_score, 2, max_seg_num
            )
        else:
            new_segs, new_scores, new_cls_idxs = NMSop.apply(
                segs, scores, cls_idxs, iou_threshold,
                min_score, max_seg_num
            )
        # seg voting
        if voting_thresh > 0:
            new_segs = seg_voting(
                new_segs,
                segs,
                scores,
                voting_thresh
            )

    # sort based on scores and return
    # truncate the results based on max_seg_num
    _, idxs = new_scores.sort(descending=True)
    max_seg_num = min(max_seg_num, new_segs.shape[0])
    # needed for multiclass NMS
    new_segs = new_segs[idxs[:max_seg_num]]
    new_scores = new_scores[idxs[:max_seg_num]]
    new_cls_idxs = new_cls_idxs[idxs[:max_seg_num]]
    return new_segs, new_scores, new_cls_idxs


def batched_nms_timestamps(
    times,          # Tensor [N]   (timestamps in feature-grid units or seconds)
    scores,         # Tensor [N]
    cls_idxs,       # Tensor [N]   (long/int)
    l1_radius,      # float        (suppress if |Δt| <= l1_radius)  -- used only in hard NMS
    min_score=0.0,  # float        (filter low-score before NMS; keep API parity)
    max_t_num=2000,
    multiclass=True,
    nms_sigma=0.5, # float or None; if set, use Gaussian Soft-NMS with this sigma (same units as `times`)
    nms_method='soft'   # 'hard' or 'soft'  -- kept for API parity; actual mode decided by nms_sigma presence
    ):
    """
    Timestamp (point) NMS with either:
      - Hard 1D NMS (default, if soft_sigma is None): suppress neighbors within |Δt| <= l1_radius
      - Gaussian Soft-NMS (if soft_sigma is not None): decay neighbors' scores as exp(-Δt^2 / (2*sigma^2))

    Kept close to original flow:
      - optional per-class NMS if multiclass=True
      - greedy keep by score order; in soft mode we do a single pass without resorting (fast Soft-NMS)
      - final global sort by score and truncate
    """
    device = times.device
    dtype_t = times.dtype
    dtype_c = cls_idxs.dtype

    num_pts = times.shape[0]
    if num_pts == 0:
        return (
            torch.zeros([0], dtype=dtype_t, device=device),
            torch.zeros([0], dtype=scores.dtype, device=device),
            torch.zeros([0], dtype=dtype_c, device=device),
        )

    # pre-filter by score (like the original does via thresholding)
    if min_score > 0:
        keep0 = scores > float(min_score)
        if not torch.all(keep0):
            times   = times[keep0]
            scores  = scores[keep0]
            cls_idxs = cls_idxs[keep0]
        num_pts = times.numel()
        if num_pts == 0:
            return (
                torch.zeros([0], dtype=dtype_t, device=device),
                torch.zeros([0], dtype=scores.dtype, device=device),
                torch.zeros([0], dtype=dtype_c, device=device),
            )

    def _nms_1d_single_class(t, s, c, radius, max_num, nms_method='hard', nms_sigma=0.5, min_score=0.0):
        # Greedy keep: sort by score desc. In soft mode, we decay later scores in-place (no re-sorting = fast Soft-NMS).
        order = torch.argsort(s, descending=True)
        t = t[order]
        s = s[order]
        c = c[order]

        if nms_method == 'hard':
            # ----- HARD NMS (unchanged from your logic) -----
            keep_mask = torch.zeros_like(order, dtype=torch.bool)
            kept_indices = []

            for i in range(t.shape[0]):
                if keep_mask[i]:
                    continue
                kept_indices.append(i)
                dt = torch.abs(t - t[i])
                keep_mask |= (dt <= radius)
                keep_mask[i] = True
                if len(kept_indices) >= max_num:
                    break

            kept_indices = torch.tensor(kept_indices, device=t.device, dtype=torch.long)
            return t[kept_indices], s[kept_indices], c[kept_indices]
        elif nms_method == 'soft':
            # ----- GAUSSIAN SOFT-NMS (1D) with re-sorting + local window -----
            if nms_sigma is None or float(nms_sigma) <= 0.0:
                return t, s, c

            sigma2 = float(nms_sigma) * float(nms_sigma)
            # Use a finite window to mimic "overlap" (IoU) neighborhood:
            # prefer user l1_radius if provided, else ~3*sigma
            cutoff = float(l1_radius) if float(l1_radius) > 0.0 else 3.0 * float(nms_sigma)

            # working copies
            t_work = t.clone()
            s_work = s.clone()

            kept_idx = []
            kept_scores = []

            # Select the current max each iteration (true Soft-NMS behavior)
            for _ in range(t_work.numel()):
                j = torch.argmax(s_work)
                # if even the top score is too small, we are done
                if s_work[j] <= float(min_score):
                    break

                # keep this item with its *current (decayed) score*
                kept_idx.append(j.item())
                kept_scores.append(s_work[j].item())
                if len(kept_idx) >= max_num:
                    break

                # decay neighbors within the local window
                dt = torch.abs(t_work - t_work[j])
                mask = (dt > 0) & (dt <= cutoff)
                if mask.any():
                    s_work[mask] = s_work[mask] * torch.exp(-(dt[mask] * dt[mask]) / (2.0 * sigma2))

                # mark selected as used so it won't be chosen again
                s_work[j] = -1e9  # sentinel << min_score

            if len(kept_idx) == 0:
                return t[:0], s[:0], c[:0]

            kept_idx = torch.tensor(kept_idx, device=t.device, dtype=torch.long)
            kept_scores = torch.tensor(kept_scores, device=s.device, dtype=s.dtype)

            # Return decayed scores (Soft-NMS outputs updated scores)
            return t[kept_idx], kept_scores, c[kept_idx]
        else:
            raise ValueError(f"Unknown nms_method: {nms_method}")

    if multiclass:
        new_times, new_scores, new_cls = [], [], []
        for class_id in torch.unique(cls_idxs):
            m = (cls_idxs == class_id)
            t_i, s_i, c_i = _nms_1d_single_class(
                times[m], scores[m], cls_idxs[m],
                l1_radius, max_t_num, nms_method=nms_method, nms_sigma=nms_sigma, min_score=min_score
            )
            if t_i.numel() > 0:
                new_times.append(t_i)
                new_scores.append(s_i)
                new_cls.append(c_i)

        new_times  = torch.cat(new_times,  dim=0) if new_times else times[:0]
        new_scores = torch.cat(new_scores, dim=0) if new_scores else scores[:0]
        new_cls    = torch.cat(new_cls,    dim=0) if new_cls else cls_idxs[:0]
    else:
        new_times, new_scores, new_cls = _nms_1d_single_class(
            times, scores, cls_idxs,
            l1_radius, max_t_num, nms_method=nms_method, nms_sigma=nms_sigma, min_score=min_score
        )

    # final sort by score and truncate (same as original)
    if new_scores.numel() == 0:
        return (
            torch.zeros([0], dtype=dtype_t, device=device),
            torch.zeros([0], dtype=scores.dtype, device=device),
            torch.zeros([0], dtype=dtype_c, device=device),
        )

    _, order = new_scores.sort(descending=True)
    max_t_num = min(max_t_num, new_scores.shape[0])
    order = order[:max_t_num]

    return new_times[order], new_scores[order], new_cls[order]