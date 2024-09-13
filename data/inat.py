import os
import json
import numpy as np
from PIL import Image
from typing import Callable, Iterator, List, Tuple, Optional
import jax
import jax.numpy as jnp
from functools import partial
from tqdm import tqdm


from concurrent.futures import ThreadPoolExecutor
import cv2  # Use OpenCV for faster image loading and processing


class INaturalist2019Dataset:
    def __init__(
            self,
            data_dir: str,
            split: str = "train",
            batch_size: int = 32,
            shuffle: bool = True,
            augmentations: Optional[Callable[[np.ndarray], np.ndarray]] = None,
            normalization: Optional[Callable[[np.ndarray], np.ndarray]] = None,
            num_workers: int = 6,
    ):
        """
        Initializes the iNaturalist2019 dataset.

        Args:
            data_dir (str): Path to the dataset directory.
            split (str): Dataset split to use ('train', 'val', or 'test').
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle the data.
            augmentations (Callable, optional): Function to apply augmentations to images.
            normalization (Callable, optional): Function to normalize images.
            num_workers (int): Number of worker threads to use for data loading.
        """
        self.data_dir = data_dir
        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentations = augmentations

        if normalization is None:
            self.normalization = self.default_normalization
        else:
            self.normalization = normalization

        self.categories = self._load_categories()
        self.category_id_to_label = {
            cat["id"]: idx for idx, cat in enumerate(self.categories)
        }
        self.image_paths, self.labels = self._load_annotations()
        self.num_samples = len(self.image_paths)
        self.num_classes = len(self.categories)
        self.indices = np.arange(self.num_samples)
        self.num_workers = num_workers

    def _load_categories(self) -> List[dict]:
        """
        Loads category information from categories.json.

        Returns:
            List[dict]: List of category dictionaries.
        """
        categories_file = os.path.join(self.data_dir, "categories.json")
        with open(categories_file, "r") as f:
            categories = json.load(f)
        return categories

    def _load_annotations(self) -> Tuple[List[str], List[int]]:
        """
        Loads annotations and constructs image paths and labels.

        Returns:
            Tuple[List[str], List[int]]: Lists of image paths and corresponding labels.
        """
        annotations_file = os.path.join(self.data_dir, f"{self.split}2019.json")
        with open(annotations_file, "r") as f:
            annotations = json.load(f)

        images = annotations["images"]
        annotations_list = annotations["annotations"]

        # Map image IDs to file names
        image_id_to_filename = {img["id"]: img["file_name"] for img in images}

        image_paths = []
        labels = []

        for ann in annotations_list:
            image_id = ann["image_id"]
            category_id = ann["category_id"]
            label = self.category_id_to_label[category_id]
            file_name = image_id_to_filename[image_id]
            image_path = os.path.join(
                self.data_dir, file_name
            )
            image_paths.append(image_path)
            labels.append(label)

        return image_paths, labels

    def __iter__(self):
        return self.generator()

    def generator(self) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Generator that yields batches of images and labels.

        Yields:
            Iterator[Tuple[jnp.ndarray, jnp.ndarray]]: Batches of images and labels.
        """
        if self.shuffle:
            np.random.shuffle(self.indices)

        for start_idx in range(0, self.num_samples, self.batch_size):
            batch_indices = self.indices[start_idx: start_idx + self.batch_size]
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                results = list(executor.map(self._load_and_process, batch_indices))
            batch_images, batch_labels = zip(*results)
            batch_images = np.stack(batch_images)
            batch_labels = np.array(batch_labels)
            # Convert to jnp arrays
            batch_images = jnp.array(batch_images)
            batch_labels = self.one_hot_encode(
                batch_labels, self.num_classes
            )
            yield batch_images, batch_labels

    def _load_and_process(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = self._load_image(image_path)
        if self.augmentations:
            image = self.augmentations(image)
        if self.normalization:
            image = self.normalization(image)
        return image, label

    def _load_image(self, image_path: str) -> np.ndarray:
        """
        Loads an image from the given path using OpenCV.

        Args:
            image_path (str): Path to the image file.

        Returns:
            np.ndarray: Loaded image as a NumPy array.
        """
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        return img

    @staticmethod
    def one_hot_encode(labels: np.ndarray, num_classes: int) -> jnp.ndarray:
        """
        One-hot encodes the labels.

        Args:
            labels (np.ndarray): Array of labels.
            num_classes (int): Number of classes.

        Returns:
            jnp.ndarray: One-hot encoded labels.
        """
        return jax.nn.one_hot(labels, num_classes)

    @staticmethod
    def default_normalization(image: np.ndarray) -> np.ndarray:
        """
        Default normalization function that scales images to [-1, 1].

        Args:
            image (np.ndarray): Image array.

        Returns:
            np.ndarray: Normalized image.
        """
        image = image / 255.0  # Scale to [0, 1]
        mean = np.array([0.485, 0.456, 0.406])  # ImageNet mean
        std = np.array([0.229, 0.224, 0.225])  # ImageNet std
        image = (image - mean) / std
        return image.astype(np.float16)  #
