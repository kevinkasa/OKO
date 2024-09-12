#!/usr/bin/env python3
# -*- coding: utf-8 -*

import os
import math
import json
import pickle
import re
from typing import Tuple, Dict, Union

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

RGB_DATASETS = ["cifar10", "cifar100", "imagenet", "imagenet_lt", "i_naturalist2018",
                'i_naturalist2018', "i_naturalist2021"]
MODELS = ["Custom", "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ViT"]

FrozenDict = config_dict.FrozenConfigDict


class UInt8orFP32(AbstractDtype):
    dtypes = ["uint8", "float32"]


class INaturalist2019Dataset:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.categories = self._load_categories()

    def _load_categories(self) -> Dict[int, str]:
        with open(os.path.join(self.data_dir, 'categories.json'), 'r') as f:
            categories = json.load(f)
        return {cat['id']: cat['name'] for cat in categories}

    def _load_split(self, split: str) -> Tuple[np.ndarray, np.ndarray]:
        json_file = f"{split}2019.json"
        with open(os.path.join(self.data_dir, json_file), 'r') as f:
            data = json.load(f)

        # Create a dictionary to map image_id to file_name
        image_id_to_file = {img['id']: img['file_name'] for img in data['images']}

        # Create a dictionary to map image_id to category_id
        image_id_to_category = {ann['image_id']: ann['category_id'] for ann in data['annotations']}

        images = []
        labels = []
        for image_id, file_name in image_id_to_file.items():
            img_path = os.path.join(self.data_dir, file_name)
            img = tf.io.read_file(img_path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [299, 299])  # Adjust size as needed
            images.append(img)

            category_id = image_id_to_category[image_id]
            labels.append(category_id)

        return np.array(images), np.array(labels)

    def load(self, split: Union[str, None] = None, batch_size: int = -1, as_supervised: bool = True) -> Union[
        tf.data.Dataset, Dict[str, tf.data.Dataset]]:
        if split is None:
            return {s: self.load(s, batch_size, as_supervised) for s in ['train', 'val', 'test']}

        images, labels = self._load_split(split)

        if batch_size == -1:
            return tf.data.Dataset.from_tensor_slices((images, labels)).batch(images.shape[0])
        else:
            ds = tf.data.Dataset.from_tensor_slices((images, labels))
            if as_supervised:
                return ds.batch(batch_size)
            else:
                return ds.map(lambda x, y: {'image': x, 'label': y}).batch(batch_size)


def get_inat_data(dataset: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    data_dir = r'/h/kkasa/datasets/inat_comp/2019/'
    ds = INaturalist2019Dataset(data_dir)
    images, labels = ds.load(split=split, batch_size=-1, as_supervised=True)
    images = jnp.asarray(images)
    labels = jax.nn.one_hot(x=labels, num_classes=np.unique(labels).shape[0])
    return (images, labels)


def convert_tf_batch_to_jax(batch):
    """
    Convert TensorFlow Tensors in a batch to JAX-compatible arrays.
    """
    images, labels = batch
    images_np = images.numpy()  # Convert TensorFlow tensor to NumPy array
    labels_np = labels.numpy()  # Convert TensorFlow tensor to NumPy array

    images_jax = jnp.asarray(images_np)  # Convert to JAX array
    labels_jax = jnp.asarray(labels_np)  # Convert to JAX array

    return images_jax, labels_jax


def create_tf_dataset(data: Tuple[Array, Array], batch_size: int,
                      shuffle: bool = False) -> tf.data.Dataset:
    """
    Converts NumPy arrays into a TensorFlow Dataset and applies batching and optional shuffling.
    """
    # Create a TensorFlow dataset from NumPy arrays
    dataset = tf.data.Dataset.from_tensor_slices((data[0], data[1]))

    # Shuffle the dataset if needed (e.g., for training data)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)

    # Batch the dataset
    dataset = dataset.batch(batch_size)

    # Prefetch to improve input pipeline performance
    # dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


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
    elif dataset == "imagenet" or 'i_naturalist2018':
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
