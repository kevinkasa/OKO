#!/usr/bin/env python3
# -*- coding: utf-8 -*

import os
import math
import time
import json
import pickle
import re
from typing import Tuple, Dict, Union, Callable, Iterator, Optional, List

import flax
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from flax import serialization
from jaxtyping import AbstractDtype, Array, Float32, jaxtyped
from ml_collections import config_dict
from typeguard import typechecked as typechecker

import torch
from torchvision import transforms
from data.inat import INatDataset, create_dataset

RGB_DATASETS = ["cifar10", "cifar100", "imagenet", "imagenet_lt", ]
MODELS = ["Custom", "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ViT"]

FrozenDict = config_dict.FrozenConfigDict


class UInt8orFP32(AbstractDtype):
    dtypes = ["uint8", "float32"]


# def build_transform():
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         # transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         normalize,
#     ])
#
#     return transform


def get_inat_data(data_dir: str, split: str, batch_size: int = 128):
    num_devices = jax.local_device_count()
    print(num_devices)
    dataset = create_dataset(
        data_dir,
        split=split,
        year=2019,
        category='name',
        batch_size=batch_size*num_devices
    )
    # # import pdb;pdb.set_trace()
    # for images, labels in dataset:
    #     start_time = time.time()  # Start the timer
    #
    #     # Simulate some processing on the batch (optional)
    #     # time.sleep(0.1)  # You can remove this line, it's just for simulation
    #
    #     end_time = time.time()  # End the timer
    #     batch_time = end_time - start_time
    #     # batch_times.append(batch_time)
    #
    #     print(f"Batch  took {batch_time} seconds to load")
    #
    # # batch_times = []  # List to store time for each batch
    # # for batch_idx, (data, labels) in enumerate(data_loader):
    # #     start_time = time.time()  # Start the timer
    # #
    # #     # Simulate some processing on the batch (optional)
    # #     # time.sleep(0.1)  # You can remove this line, it's just for simulation
    # #
    # #     end_time = time.time()  # End the timer
    # #     batch_time = end_time - start_time
    # #     # batch_times.append(batch_time)
    # #
    # #     print(f"Batch {batch_idx + 1} took {batch_time} seconds to load")
    # #
    # # average_batch_time = sum(batch_times) / len(batch_times)
    # # print(f"Average time per batch: {average_batch_time:.4f} seconds")
    # # import pdb;pdb.set_trace()
    return dataset


def convert_batch_to_jax(batch):
    """
    Convert TensorFlow Tensors in a batch to JAX-compatible arrays.
    """
    images, labels = batch
    images_np = images.numpy()  # Convert TensorFlow tensor to NumPy array
    labels_np = labels.numpy()  # Convert TensorFlow tensor to NumPy array

    images_jax = jnp.asarray(images_np)  # Convert to JAX array
    labels_jax = jnp.asarray(labels_np)  # Convert to JAX array

    return images_jax, labels_jax


# def create_tf_dataset(data: Tuple[Array, Array], batch_size: int,
#                       shuffle: bool = False) -> tf.data.Dataset:
#     """
#     Converts NumPy arrays into a TensorFlow Dataset and applies batching and optional shuffling.
#     """
#     # Create a TensorFlow dataset from NumPy arrays
#     dataset = tf.data.Dataset.from_tensor_slices((data[0], data[1]))
#
#     # Shuffle the dataset if needed (e.g., for training data)
#     if shuffle:
#         dataset = dataset.shuffle(buffer_size=10000)
#
#     # Batch the dataset
#     dataset = dataset.batch(batch_size, drop_remainder=True)
#
#     # Prefetch to improve input pipeline performance
#     # dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
#
#     return dataset


def get_data(dataset: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    tf_split = get_tf_split(split)
    images, labels = tfds.as_numpy(
        tfds.load(
            dataset,
            split=tf_split,
            batch_size=-1,
            as_supervised=True,
            data_dir=r'/scratch/ssd004/scratch/kkasa/data',
            download=True,
        )
    )
    images = jnp.asarray(images)
    labels = jax.nn.one_hot(x=labels, num_classes=np.unique(labels).shape[0])
    return (images, labels)


def get_tf_split(split: str) -> str:
    if split == "train":
        tf_split = "train[:80%]"
    elif split == "val":
        tf_split = "train[80%:]"
    else:
        tf_split = split
    return tf_split


def get_data_statistics(
        dataset: str,
) -> Tuple[Float32[Array, "3"], Float32[Array, "3"]]:
    """Get means and stds of CIFAR-10, CIFAR-100, or the ImageNet training data."""
    if dataset == "cifar10":
        means = jnp.array([0.4914, 0.4822, 0.4465], dtype=jnp.float32)
        stds = jnp.array([0.2023, 0.1994, 0.2010], dtype=jnp.float32)
    elif dataset == "cifar100":
        means = jnp.array([0.5071, 0.4865, 0.44092], dtype=jnp.float32)
        stds = jnp.array([0.2673, 0.2564, 0.2761], dtype=jnp.float32)
    elif dataset == "imagenet" or 'i_naturalist2019':
        means = jnp.array([0.485, 0.456, 0.406], dtype=jnp.float32)
        stds = jnp.array([0.229, 0.224, 0.225], dtype=jnp.float32)
    else:
        raise Exception(f"\nDataset statistics for {dataset} are not available.\n")
    return means, stds


@jaxtyped
@typechecker
def normalize_images(
        images: UInt8orFP32[Array, "#batchk h w c"],
        data_config: FrozenDict,
) -> UInt8orFP32[Array, "#batchk h w c"]:
    images = images / data_config.max_pixel_value
    images -= data_config.means
    images /= data_config.stds
    return images


def load_metrics(metric_path):
    """Load pretrained parameters into memory."""
    binary = find_binaries(metric_path)
    metrics = pickle.loads(open(os.path.join(metric_path, binary), "rb").read())
    return metrics


def save_params(out_path, params, epoch):
    """Encode parameters of network as bytes and save as binary file."""
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
    bytes_output = serialization.to_bytes(params)
    with open(os.path.join(out_path, f"pretrained_params_{epoch}.pkl"), "wb") as f:
        pickle.dump(bytes_output, f)


def save_opt_state(out_path, opt_state, epoch):
    """Encode parameters of network as bytes and save as binary file."""
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
    bytes_output = serialization.to_bytes(opt_state)
    with open(os.path.join(out_path, f"opt_state_{epoch}.pkl"), "wb") as f:
        pickle.dump(bytes_output, f)


def find_binaries(param_path):
    """Search for last checkpoint."""
    param_binaries = sorted(
        [
            f
            for _, _, files in os.walk(param_path)
            for f in files
            if re.search(r"(?=.*\d+)(?=.*pkl$)", f)
        ]
    )
    return param_binaries.pop()


def merge_params(pretrained_params, current_params):
    return flax.core.FrozenDict(
        {"encoder": pretrained_params["encoder"], "clf": current_params["clf"]}
    )


def get_subset(y, hist):
    subset = []
    for k, freq in enumerate(hist):
        subset.extend(
            np.random.choice(np.where(y == k)[0], size=freq, replace=False).tolist()
        )
    subset = np.random.permutation(subset)
    return subset
