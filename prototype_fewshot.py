import torch
import torch.nn.functional as F


def sample_multilabel_kshot(labels, active_indices, shots_per_class, seed):
    """Select exactly K positive supervision examples for every active class.

    EMOTIC is multi-label. The returned subset is the union of K independently
    sampled positive anchors per class. A positive co-label is ignored unless
    that sample was selected as an anchor for that class; true negatives in the
    union remain supervised. This prevents label overlap from silently turning
    a K-shot run into a greater-than-K-shot run.
    """
    if labels.ndim != 2:
        raise ValueError("labels must have shape [samples, classes]")
    if shots_per_class <= 0:
        raise ValueError("shots_per_class must be positive")

    active_indices = torch.as_tensor(active_indices, dtype=torch.long)
    if active_indices.numel() == 0:
        raise ValueError("active_indices must not be empty")
    generator = torch.Generator().manual_seed(int(seed))
    selected_by_class = {}
    selected_union = set()

    for class_id in active_indices.tolist():
        positive_indices = torch.nonzero(
            labels[:, class_id].gt(0), as_tuple=False
        ).flatten()
        available = int(positive_indices.numel())
        if available < shots_per_class:
            raise ValueError(
                f"Class {class_id} has {available} positive samples, fewer "
                f"than requested K={shots_per_class}"
            )
        order = torch.randperm(available, generator=generator)
        chosen = positive_indices[order[:shots_per_class]].tolist()
        selected_by_class[class_id] = chosen
        selected_union.update(chosen)

    selected_indices = torch.tensor(
        sorted(selected_union), dtype=torch.long
    )
    subset_labels = labels[selected_indices]
    # All genuine negatives in the selected image union are useful supervision.
    # Positive co-labels are masked unless explicitly selected for that class.
    supervision_mask = subset_labels.eq(0)
    global_to_local = {
        int(global_id): local_id
        for local_id, global_id in enumerate(selected_indices.tolist())
    }
    for class_id, chosen in selected_by_class.items():
        local_ids = [global_to_local[int(index)] for index in chosen]
        supervision_mask[local_ids, class_id] = True

    rows = []
    for class_id in active_indices.tolist():
        class_mask = supervision_mask[:, class_id]
        class_labels = subset_labels[:, class_id].bool()
        rows.append(
            {
                "class_id": class_id,
                "available_positives": int(labels[:, class_id].sum().item()),
                "selected_positive_anchors": shots_per_class,
                "supervised_positives": int(
                    (class_mask & class_labels).sum().item()
                ),
                "ignored_positive_colabels": int(
                    ((~class_mask) & class_labels).sum().item()
                ),
                "supervised_negatives": int(
                    (class_mask & ~class_labels).sum().item()
                ),
            }
        )
    return selected_indices, supervision_mask, rows


def masked_pos_weight(labels, supervision_mask, active_indices, max_weight=20.0):
    active_indices = torch.as_tensor(active_indices, dtype=torch.long)
    active_labels = labels[:, active_indices]
    active_mask = supervision_mask[:, active_indices].bool()
    positives = (active_labels.gt(0) & active_mask).sum(dim=0).float()
    negatives = (active_labels.eq(0) & active_mask).sum(dim=0).float()
    return (negatives / positives.clamp_min(1.0)).clamp(max=max_weight)


def masked_bce_with_logits(
    logits,
    targets,
    supervision_mask,
    pos_weight=None,
):
    losses = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weight,
        reduction="none",
    )
    mask = supervision_mask.to(dtype=losses.dtype)
    denominator = mask.sum()
    if denominator.item() <= 0:
        raise ValueError("supervision_mask contains no supervised entries")
    return (losses * mask).sum() / denominator
