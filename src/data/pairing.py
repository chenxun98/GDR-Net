from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CounterfactualPairs:
    pair_index: torch.Tensor
    mask: torch.Tensor
    cls_mask: torch.Tensor


def _choose(candidates: list[int], generator: torch.Generator | None) -> int:
    if len(candidates) == 1:
        return candidates[0]
    index = int(torch.randint(len(candidates), (1,), generator=generator).item())
    return candidates[index]


def build_counterfactual_pairs(
    batch: dict,
    task: str,
    generator: torch.Generator | None = None,
) -> CounterfactualPairs:
    """Build label-compatible, different-patient recombination pairs.

    Cross-domain donors are preferred. If no compatible donor exists, the
    anchor points to itself only as a safe tensor index and its masks are zero.
    """

    task = str(task).lower()
    if task not in {"detection", "subtype", "subtype_classification", "classification", "multiclass"}:
        raise ValueError(f"Unsupported counterfactual pairing task: {task!r}")

    patient_ids = [str(value) for value in batch["patient_id"]]
    domains = [int(value) for value in batch["domain_label"].tolist()]
    det_labels = [int(value) for value in batch["det_label"].tolist()]
    cls_labels = [int(value) for value in batch["cls_label"].tolist()]
    is_subtype = task != "detection"

    pair_index: list[int] = []
    masks: list[float] = []
    cls_masks: list[float] = []

    for anchor, patient_id in enumerate(patient_ids):
        compatible: list[int] = []
        for donor, donor_patient_id in enumerate(patient_ids):
            if donor_patient_id == patient_id:
                continue
            if det_labels[donor] != det_labels[anchor]:
                continue
            if is_subtype and det_labels[anchor] == 1:
                if cls_labels[anchor] < 0 or cls_labels[donor] != cls_labels[anchor]:
                    continue
            compatible.append(donor)

        cross_domain = [donor for donor in compatible if domains[donor] != domains[anchor]]
        candidates = cross_domain or compatible
        if not candidates:
            pair_index.append(anchor)
            masks.append(0.0)
            cls_masks.append(0.0)
            continue

        donor = _choose(candidates, generator)
        pair_index.append(donor)
        masks.append(1.0)
        cls_masks.append(
            1.0
            if is_subtype
            and det_labels[anchor] == 1
            and cls_labels[anchor] >= 0
            and cls_labels[donor] == cls_labels[anchor]
            else 0.0
        )

    return CounterfactualPairs(
        pair_index=torch.tensor(pair_index, dtype=torch.long),
        mask=torch.tensor(masks, dtype=torch.float32),
        cls_mask=torch.tensor(cls_masks, dtype=torch.float32),
    )
