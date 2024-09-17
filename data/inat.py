import os
import json
from torchvision.datasets.folder import ImageFolder, default_loader
import tensorflow as tf

# class INatDataset(ImageFolder):
#     def __init__(self, root, split='train', year=2018, transform=None, target_transform=None, category='name',
#                  loader=default_loader):
#         super().__init__(root, transform, target_transform, loader)
#         self.transform = transform
#         self.loader = loader
#         self.target_transform = target_transform
#         self.year = year
#         # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
#         path_json = os.path.join(root, f'{split}{year}.json')
#         with open(path_json) as json_file:
#             data = json.load(json_file)
#
#         with open(os.path.join(root, 'categories.json')) as json_file:
#             data_catg = json.load(json_file)
#
#         path_json_for_targeter = os.path.join(root, f"train{year}.json")
#
#         with open(path_json_for_targeter) as json_file:
#             data_for_targeter = json.load(json_file)
#
#         targeter = {}
#         indexer = 0
#         for elem in data_for_targeter['annotations']:
#             king = []
#             king.append(data_catg[int(elem['category_id'])][category])
#             if king[0] not in targeter.keys():
#                 targeter[king[0]] = indexer
#                 indexer += 1
#         self.nb_classes = len(targeter)
#
#         self.samples = []
#         for elem in data['images']:
#             cut = elem['file_name'].split('/')
#             target_current = int(cut[2])
#             path_current = os.path.join(root, cut[0], cut[1], cut[2], cut[3])
#
#             categors = data_catg[target_current]
#             target_current_true = targeter[categors[category]]
#             self.samples.append((path_current, target_current_true))
#             # import pdb;pdb.set_trace()
#     # __getitem__ and __len__ inherited from ImageFolder


from tensorflow.keras.preprocessing import image


class INatDataset:
    def __init__(self, root, split='train', year=2018, category='name'):
        self.root = root
        self.split = split
        self.year = year
        self.category = category

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

    def get_dataset(self):
        paths, labels = zip(*self.samples)
        dataset = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))
        return dataset


@tf.function
def load_and_preprocess_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


@tf.function
def normalize_image(image, label):
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    image = (image - mean) / std
    return image, label


def create_dataset(data_dir, split, year, category, batch_size):
    dataset = INatDataset(data_dir, split=split, year=year, category=category)
    ds = dataset.get_dataset()

    # Apply transformations
    ds = ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(normalize_image, num_parallel_calls=tf.data.AUTOTUNE)

    if split == 'train':
        ds = ds.shuffle(buffer_size=10000)

    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds
