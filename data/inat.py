from collections import defaultdict
import os
import json
import tensorflow as tf
import numpy as np

class INatDataset:
    def __init__(self, root, split='train', year=2018, category='name', set_size=3):
        self.root = root
        self.split = split
        self.year = year
        self.category = category


        self.set_size = set_size
        self.num_odds = set_size - 2  # Always 1 in your example, but kept flexible

        path_json = os.path.join(root, f'{split}{year}.json')
        with open(path_json) as json_file:
            self.data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            self.data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")
        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        self.targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = self.data_catg[int(elem['category_id'])][category]
            if king not in self.targeter.keys():
                self.targeter[king] = indexer
                indexer += 1

        self.nb_classes = len(self.targeter)

        self.samples = []
        for elem in self.data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[1], cut[2], cut[3])
            categors = self.data_catg[target_current]
            target_current_true = self.targeter[categors[category]]
            self.samples.append((path_current, target_current_true))
        # Group samples by class
        self.samples_by_class = defaultdict(list)
        for path, label in self.samples:
            self.samples_by_class[label].append(path)

    def get_dataset(self):
        paths, labels = zip(*self.samples)
        dataset = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))
        return dataset

    def make_sets(self):
        all_classes = list(self.samples_by_class.keys())

        def generate_sets():
            # Shuffle samples within each class
            for class_label in self.samples_by_class:
                np.random.shuffle(self.samples_by_class[class_label])
            import pdb;pdb.set_trace()
            # Create a set for each sample
            for pair_class in all_classes:
                for pair_sample in self.samples_by_class[pair_class]:
                    set_samples = [pair_sample]

                    # Add another sample from the same class
                    same_class_samples = [s for s in self.samples_by_class[pair_class] if s != pair_sample]
                    if same_class_samples:
                        set_samples.append(np.random.choice(same_class_samples))
                    else:
                        # If no other samples in the same class, choose from a different class
                        other_class = np.random.choice([c for c in all_classes if c != pair_class])
                        set_samples.append(np.random.choice(self.samples_by_class[other_class]))

                    # Add samples from different classes for the odd ones
                    other_classes = [c for c in all_classes if c != pair_class]
                    for _ in range(self.set_size - 2):
                        odd_class = np.random.choice(other_classes)
                        set_samples.append(np.random.choice(self.samples_by_class[odd_class]))
                        other_classes = [c for c in other_classes if c != odd_class]

                    # Yield paths and target
                    yield set_samples, pair_class

        return tf.data.Dataset.from_generator(
            generate_sets,
            output_signature=(
                tf.TensorSpec(shape=(self.set_size,), dtype=tf.string),
                tf.TensorSpec(shape=(), dtype=tf.int32)
            )
        )


@tf.function
def load_and_preprocess_image(paths, target):
    def process_single_image(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [224, 224])
        img = tf.cast(img, tf.float32) / 255.0
        return img

    images = tf.map_fn(process_single_image, paths, fn_output_signature=tf.float32)
    return images, target


@tf.function
def normalize_image(images, target):
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    images = (images - mean) / std
    return images, target


def create_dataset(data_dir, split, year, category, batch_size, set_size=3):
    dataset = INatDataset(data_dir, split=split, year=year, category=category, set_size=set_size)

    if split == 'train':
        ds = dataset.make_sets()
    else:
        ds = dataset.get_dataset()
        # For validation/test, we need to ensure the data is in the right shape
        ds = ds.map(lambda path, label: (tf.expand_dims(path, 0), label))

    ds = ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(normalize_image, num_parallel_calls=tf.data.AUTOTUNE)

    if split != 'train':
        # Remove the extra dimension we added for validation/test
        ds = ds.map(lambda images, target: (tf.squeeze(images), target))

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
