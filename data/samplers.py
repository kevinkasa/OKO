from collections import defaultdict, Counter
import random
import pdb
import numpy as np

from torch.utils.data import Sampler, BatchSampler
import itertools
from typing import (
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Sized,
    TypeVar,
    Union,
)

import torch

class BalancedBatchSamplerV3(BatchSampler):
    """
    BatchSampler that returns batches of size batch_size.
    Each batch contains at least 2 samples per class.
    Data indices are used only once, and all data indices are returned
    (except possibly in the last incomplete batch).
    """

    def __init__(self, labels, batch_size, drop_last=True):
        self.labels = np.array(labels)
        self.labels_set = list(set(self.labels))
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        # Re-initialize label_to_indices and shuffle data at the start of each epoch
        label_to_indices = {
            label: np.where(self.labels == label)[0].tolist()
            for label in self.labels_set
        }
        # Shuffle indices within each class
        for label in self.labels_set:
            np.random.shuffle(label_to_indices[label])

        # Prepare per-class index iterators
        class_iters = {label: iter(indices) for label, indices in label_to_indices.items()}
        # Initialize the list of classes with available samples
        available_classes = set(self.labels_set)

        batch = []
        while True:
            # If batch is full, yield it
            if len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                batch = batch[self.batch_size:]

            # If no classes have available samples, break
            if not available_classes:
                if batch and not self.drop_last:
                    yield batch
                break

            # Randomly select a class with available samples
            selected_class = np.random.choice(list(available_classes))
            indices = []
            num_samples_needed = min(2, self.batch_size - len(batch))

            # Try to get at least 2 samples from this class
            for _ in range(num_samples_needed):
                try:
                    index = next(class_iters[selected_class])
                    indices.append(index)
                except StopIteration:
                    # No more samples in this class
                    available_classes.remove(selected_class)
                    break  # Exit the loop to select a new class

            if indices:
                batch.extend(indices)

            # If we couldn't get 2 samples, try another class
            if len(indices) < 2 and available_classes:
                continue

        # Yield any remaining samples if drop_last is False
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        total_samples = len(self.labels)
        if self.drop_last:
            return total_samples // self.batch_size
        else:
            return (total_samples + self.batch_size - 1) // self.batch_size

class BalancedBatchSamplerV2(BatchSampler):
    """
    BatchSampler - samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples, drop_last=True):
        self.labels = labels
        self.labels_set = list(set(self.labels))
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.batch_size = n_classes * n_samples
        self.drop_last = drop_last

        # Prepare per-class index queues
        self.label_to_indices = {
            label: np.where(np.asarray(labels) == label)[0].tolist()
            for label in self.labels_set
        }
        for label in self.labels_set:
            np.random.shuffle(self.label_to_indices[label])

        # Create per-class chunks
        self.class_chunks = {}
        for label in self.labels_set:
            indices = self.label_to_indices[label]
            chunks = [
                indices[i: i + n_samples] for i in range(0, len(indices), n_samples)
            ]
            # Remove the last chunk if it's smaller than n_samples and drop_last is True
            if len(chunks[-1]) < n_samples:
                if self.drop_last:
                    chunks = chunks[:-1]
            self.class_chunks[label] = chunks

        # Initialize the list of labels that have available chunks
        self.labels_with_chunks = [
            label for label in self.labels_set if len(self.class_chunks[label]) > 0
        ]

    def __iter__(self):
        # We will keep picking batches until no more chunks are available
        while len(self.labels_with_chunks) >= self.n_classes:
            # Randomly select n_classes from labels_with_chunks
            selected_labels = np.random.choice(
                self.labels_with_chunks, self.n_classes, replace=False
            )
            batch = []
            for label in selected_labels:
                # Pop the first chunk from the class
                chunk = self.class_chunks[label].pop(0)
                batch.extend(chunk)
                # If the class has no more chunks, remove it from labels_with_chunks
                if len(self.class_chunks[label]) == 0:
                    self.labels_with_chunks.remove(label)
            # Ensure the batch is of the correct size
            if len(batch) == self.batch_size:
                yield batch
            else:
                if not self.drop_last and len(batch) > 0:
                    yield batch
                # else, discard the incomplete batch

    def __len__(self):
        # Calculate the number of batches we can produce
        total_chunks = sum(len(chunks) for chunks in self.class_chunks.values())
        num_batches = total_chunks // self.n_classes
        return num_batches


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: np.where(np.asarray(self.labels) == label)[0].tolist() for label in
                                 self.labels_set}

        # Track used indices per class
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.count = 0
        self.batch_size = n_classes * n_samples

    def _reset(self):
        self.label_to_indices = {label: np.where(np.asarray(self.labels) == label)[0].tolist() for label in
                                 self.labels_set}

    def __iter__(self):
        self._reset()
        self.count = 0
        used_indices = set()
        _label_set = set(self.labels)

        while self.count + self.batch_size < self.n_dataset:
            if len(_label_set) >= self.n_classes:
                classes = np.random.choice(list(_label_set), self.n_classes, replace=False)
            elif len(_label_set) < self.n_classes and len(_label_set) > 0:
                classes = np.random.choice(list(_label_set), self.n_classes, replace=True)
            else:
                # pdb.set_trace()
                self.count += len(indices)
                if len(indices) == 0:
                    break
                else:
                    yield indices

            indices = []
            for class_ in classes:
                available_indices = self.label_to_indices[class_]

                # Check if enough samples are available for this class
                if len(available_indices) < self.n_samples:
                    if len(available_indices) >= 2:
                        selected_indices = available_indices[:]
                        # pdb.set_trace()
                        a = 1
                    else:
                        # pdb.set_trace()
                        # pdb.set_trace()
                        if class_ in _label_set:
                            _label_set.remove(class_)
                        # if len(list(_label_set - set(classes))) >= 1:
                        #     new_class = np.random.choice(list(_label_set - set(classes)), 1, replace=False)
                        #     classes = np.append(classes, new_class)
                        continue  # Skip this class if fewer than 2 samples are left
                else:
                    selected_indices = available_indices[:self.n_samples]

                # Remove the selected indices from the class pool to avoid reuse
                self.label_to_indices[class_] = available_indices[len(selected_indices):]
                indices.extend(selected_indices)

            # Add only new indices to the batch
            indices = [idx for idx in indices if idx not in used_indices]
            used_indices.update(indices)

            # Ensure valid batch size
            if len(indices) >= 2:
                # if len(indices) != 128:
                #     a = 1
                #     #loss pdb.set_trace()
                #     b = 1

                self.count += len(indices)
                yield indices

    def __len__(self):
        return self.n_dataset // (self.n_samples * self.n_classes)


class BatchSampler(Sampler[List[int]]):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(
            self,
            sampler: Union[Sampler[int], Iterable[int]],
            batch_size: int,
            drop_last: bool,
    ) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if (
                not isinstance(batch_size, int)
                or isinstance(batch_size, bool)
                or batch_size <= 0
        ):
            raise ValueError(
                f"batch_size should be a positive integer value, but got batch_size={batch_size}"
            )
        if not isinstance(drop_last, bool):
            raise ValueError(
                f"drop_last should be a boolean value, but got drop_last={drop_last}"
            )
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        # Implemented based on the benchmarking in https://github.com/pytorch/pytorch/pull/76951
        sampler_iter = iter(self.sampler)
        pdb.set_trace()
        if self.drop_last:
            # Create multiple references to the same iterator
            args = [sampler_iter] * self.batch_size

            for batch_droplast in zip(*args):
                yield [*batch_droplast]
        else:
            batch = [*itertools.islice(sampler_iter, self.batch_size)]
            while batch:
                yield batch
                batch = [*itertools.islice(sampler_iter, self.batch_size)]

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore[arg-type]
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]


class EfficientMinSamplesPerClassSampler(Sampler):
    """
    An efficient sampler that maintains the original data distribution as much as possible
    while ensuring that any class present in a batch has at least a minimum number of samples.
    Avoids reusing indices across batches and excludes classes that can't meet the requirement.
    Supports shuffle functionality similar to PyTorch DataLoader.
    """

    def __init__(self, labels, batch_size, min_samples_per_class=2, shuffle=False, drop_last=False):
        """
        Args:
            labels (list): List of class labels for each sample in the dataset.
            batch_size (int): Desired batch size.
            min_samples_per_class (int): Minimum number of samples per class in a batch.
            shuffle (bool): Set to True to have the data reshuffled at every epoch.
            drop_last (bool): Set to True to drop the last incomplete batch.
        """
        self.labels = labels
        self.batch_size = batch_size
        self.min_samples_per_class = min_samples_per_class
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Map labels to indices
        self.label_to_indices = defaultdict(list)
        pdb.set_trace()
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)

        # Exclude classes that have fewer samples than min_samples_per_class
        self.valid_labels = [
            label for label, indices in self.label_to_indices.items()
            if len(indices) >= self.min_samples_per_class
        ]
        # Remove invalid labels from label_to_indices
        self.label_to_indices = {
            label: indices for label, indices in self.label_to_indices.items()
            if label in self.valid_labels
        }

        # Total number of valid samples
        self.num_samples = sum(len(indices) for indices in self.label_to_indices.values())

        # Initialize the internal state
        self.reset()

    def reset(self):
        # Shuffle indices within each class if shuffle is True
        if self.shuffle:
            for label in self.label_to_indices:
                random.shuffle(self.label_to_indices[label])

        # Reset positions
        self.label_to_position = {label: 0 for label in self.label_to_indices}

        # Create a list of labels to sample from
        self.all_labels = list(self.valid_labels)
        if self.shuffle:
            random.shuffle(self.all_labels)

        # Reset used indices
        self.used_indices = set()

    def __iter__(self):
        self.reset()  # Reset at the beginning of each epoch
        pdb.set_trace()
        batch_indices = []
        class_counts_in_batch = defaultdict(int)
        available_labels = set(self.all_labels)

        while True:
            if not available_labels:
                # No labels can provide more samples
                if batch_indices:
                    if not self.drop_last or len(batch_indices) == self.batch_size:
                        yield batch_indices[:self.batch_size]
                break  # Exit the loop

            for label in list(available_labels):
                indices_of_label = self.label_to_indices[label]
                position = self.label_to_position[label]

                # Check if enough samples remain for this label
                remaining_samples = len(indices_of_label) - position
                if remaining_samples < self.min_samples_per_class:
                    available_labels.remove(label)
                    continue  # Not enough samples left for this class

                # Add min_samples_per_class samples of this label
                samples_added = 0
                while samples_added < self.min_samples_per_class and position < len(indices_of_label):
                    idx_to_add = indices_of_label[position]
                    if idx_to_add not in self.used_indices:
                        batch_indices.append(idx_to_add)
                        self.used_indices.add(idx_to_add)
                        class_counts_in_batch[label] += 1
                        samples_added += 1
                    position += 1

                # Update the position for this label
                self.label_to_position[label] = position

                # If batch is full, yield it
                if len(batch_indices) >= self.batch_size:
                    yield batch_indices[:self.batch_size]
                    batch_indices = []
                    class_counts_in_batch = defaultdict(int)
                    if self.shuffle:
                        random.shuffle(self.all_labels)
                    available_labels = set(self.all_labels)
                    break  # Break to start filling the next batch

                # If this label cannot provide more samples in future iterations
                if (len(indices_of_label) - self.label_to_position[label]) < self.min_samples_per_class:
                    available_labels.remove(label)

            else:
                # After going through all labels, check if batch can be yielded
                if batch_indices:
                    if not self.drop_last or len(batch_indices) == self.batch_size:
                        yield batch_indices[:self.batch_size]
                    batch_indices = []
                    class_counts_in_batch = defaultdict(int)
                    if not available_labels:
                        break  # Exit the loop
                else:
                    # No samples added, and no available labels
                    break  # Exit the loop

    def __len__(self):
        # Calculate the total number of batches
        total_batches = 0
        total_samples = self.num_samples
        batch_size = self.batch_size

        # Compute the number of full batches
        total_batches = total_samples // batch_size

        # If there are leftover samples and drop_last is False, add one more batch
        if not self.drop_last and total_samples % batch_size != 0:
            total_batches += 1

        return total_batches
