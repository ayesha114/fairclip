"""
=============================================================================
Demographic-Balanced Sampler
=============================================================================
Custom PyTorch sampler that constructs each batch with EQUAL representation
across demographic groups.

WHY THIS IS CRITICAL FOR FAIRCLIP:
The fairness regularizer (Step VI of the methodology) penalizes group-variance
in embeddings. If a batch contains 60 male faces and 4 female faces, the
group-variance estimate is dominated by within-male variance and the
fairness signal collapses.

Random sampling on FairFace (~50/50 gender split) doesn't guarantee per-batch
balance — even a perfectly balanced dataset gives ~5-10% imbalance per batch
by chance. With 7-way race or 9-way age, random batches can be wildly skewed.

This sampler guarantees that, in every batch of size B, each of the K groups
contributes exactly B/K samples. If B is not divisible by K, the closest
divisible size is used.

Reference: this is the same approach used by Liu et al. (NeurIPS 2023) and
Zhang et al. CVPR 2025 for their joint VL training.
=============================================================================
"""

import logging
import math
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler

log = logging.getLogger(__name__)


class DemographicBalancedSampler(Sampler[int]):
    """
    Sampler that yields indices such that each batch has equal representation
    of each demographic group.

    Args:
        labels: length-N sequence of integer group labels (e.g. gender_idx for
                every row in the dataset).
        batch_size: target batch size. Must be >= num_groups.
        num_batches: number of batches per epoch. If None, uses
                     ceil(N / batch_size).
        seed: random seed for reproducibility (changes per epoch via set_epoch).
        drop_last: if True, drops indices that don't fit cleanly into a batch.
                   Recommended True for training; False for evaluation.

    Yields:
        Integer indices into the dataset, in batches of size B with
        balanced groups.

    Example:
        >>> dataset = FaceDataset("manifest.csv")
        >>> labels = dataset.df["gender_idx"].values
        >>> sampler = DemographicBalancedSampler(labels, batch_size=64)
        >>> loader = DataLoader(dataset, batch_sampler=sampler.batches())

    Note: This is a SAMPLER (yields indices) and also exposes a .batches()
    method for use as a batch_sampler. Both interfaces are provided because
    DataLoader accepts either.
    """

    def __init__(
        self,
        labels: Sequence[int] | np.ndarray | torch.Tensor,
        batch_size: int,
        num_batches: int | None = None,
        seed: int = 42,
        drop_last: bool = True,
    ):
        # Convert labels to numpy for vectorized operations
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        labels = np.asarray(labels)

        # Find the unique groups, IGNORING the missing-data sentinel -1.
        # Items with label=-1 are excluded from the sampler entirely.
        valid_mask = labels >= 0
        if not valid_mask.any():
            raise ValueError("All labels are -1 (missing); sampler has nothing to yield.")
        if (~valid_mask).any():
            n_invalid = (~valid_mask).sum()
            log.warning(f"DemographicBalancedSampler: excluding {n_invalid} items with label=-1")

        self.labels = labels
        self.valid_indices = np.where(valid_mask)[0]
        self.unique_groups = np.unique(labels[valid_mask])
        self.num_groups = len(self.unique_groups)

        if batch_size < self.num_groups:
            raise ValueError(
                f"batch_size ({batch_size}) must be >= num_groups ({self.num_groups}). "
                f"Otherwise we cannot put one sample from each group in a batch."
            )

        # Round batch_size down to the nearest multiple of num_groups so that
        # each group contributes exactly per_group samples. The trainer should
        # be aware of this — we log it loudly.
        self.per_group = batch_size // self.num_groups
        self.effective_batch_size = self.per_group * self.num_groups
        if self.effective_batch_size != batch_size:
            log.warning(
                f"batch_size {batch_size} not divisible by {self.num_groups} groups; "
                f"using effective batch size {self.effective_batch_size} "
                f"({self.per_group} per group)."
            )

        # Group indices by label, restricted to valid items
        self.group_to_indices: dict[int, np.ndarray] = {}
        for g in self.unique_groups:
            self.group_to_indices[int(g)] = np.where(labels == g)[0]
            log.info(
                f"  Group {g}: {len(self.group_to_indices[int(g)])} samples "
                f"({100*len(self.group_to_indices[int(g)])/len(labels):.1f}%)"
            )

        # Determine number of batches per epoch.
        # We use the smallest group's count divided by per_group as the limit,
        # because once we exhaust the smallest group, we'd need to repeat samples.
        smallest_group_size = min(len(idxs) for idxs in self.group_to_indices.values())
        max_batches = smallest_group_size // self.per_group

        if num_batches is None:
            self.num_batches = max_batches
        else:
            if num_batches > max_batches:
                log.warning(
                    f"Requested {num_batches} batches but smallest group only "
                    f"supports {max_batches}. Sampling WITH replacement to fill."
                )
            self.num_batches = num_batches

        log.info(
            f"DemographicBalancedSampler: {self.num_groups} groups, "
            f"{self.per_group}/group, {self.num_batches} batches/epoch, "
            f"effective batch size {self.effective_batch_size}"
        )

        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0  # Used to vary the random seed across epochs

    def set_epoch(self, epoch: int) -> None:
        """Change the random seed for a new epoch (used by DistributedSampler convention)."""
        self.epoch = epoch

    def _generate_batches(self) -> list[list[int]]:
        """Build all batches for one epoch (called fresh each epoch)."""
        rng = np.random.default_rng(seed=self.seed + self.epoch)

        # For each group, shuffle its indices. We use sample-WITH-replacement
        # only when the requested number of batches exceeds what the smallest
        # group can supply without repetition.
        smallest_group_size = min(len(idxs) for idxs in self.group_to_indices.values())
        max_batches_no_replacement = smallest_group_size // self.per_group
        with_replacement = self.num_batches > max_batches_no_replacement

        # For each group, build a queue of indices long enough to serve all batches.
        group_queues: dict[int, list[int]] = {}
        needed_per_group = self.num_batches * self.per_group
        for g, idxs in self.group_to_indices.items():
            if with_replacement:
                # Sample with replacement to fill the queue
                queue = rng.choice(idxs, size=needed_per_group, replace=True)
            else:
                # Shuffle and take the first needed_per_group items
                shuffled = rng.permutation(idxs)
                queue = shuffled[:needed_per_group]
            group_queues[g] = queue.tolist()

        # Build batches by drawing per_group items from each group queue
        batches = []
        for batch_idx in range(self.num_batches):
            batch = []
            for g in self.unique_groups:
                start = batch_idx * self.per_group
                end = start + self.per_group
                batch.extend(group_queues[int(g)][start:end])
            # Shuffle within the batch so the order of items isn't always
            # group0, group0, ..., group1, group1, ...
            rng.shuffle(batch)
            batches.append(batch)

        return batches

    def __iter__(self) -> Iterator[int]:
        """Yield individual indices flattened across all batches in the epoch."""
        for batch in self._generate_batches():
            for idx in batch:
                yield int(idx)

    def __len__(self) -> int:
        """Total number of indices yielded per epoch."""
        return self.num_batches * self.effective_batch_size

    def batches(self) -> "BatchView":
        """
        Return a batch-view object that can be used as DataLoader's batch_sampler.

        Example:
            sampler = DemographicBalancedSampler(labels, batch_size=64)
            loader = DataLoader(dataset, batch_sampler=sampler.batches())
        """
        return BatchView(self)


class BatchView:
    """Adapter that exposes DemographicBalancedSampler as a batch_sampler."""

    def __init__(self, parent: DemographicBalancedSampler):
        self.parent = parent

    def __iter__(self) -> Iterator[list[int]]:
        for batch in self.parent._generate_batches():
            yield batch

    def __len__(self) -> int:
        return self.parent.num_batches

    def set_epoch(self, epoch: int) -> None:
        self.parent.set_epoch(epoch)
