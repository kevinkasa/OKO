import os
import json
from PIL import Image
from torchvision import transforms
from torchvision.datasets.folder import ImageFolder, default_loader
import pdb
from tqdm import tqdm


class INatDataset(ImageFolder):
    def __init__(self, root, split='train', year=2018, category='name', transform=None, k=1, loader=default_loader,
                 target_transform=None):
        self.root = root
        self.split = split
        self.year = year
        self.category = category
        self.transform = transform
        self.k = k
        self.loader = loader
        self.target_transform = target_transform

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
        self.samples_by_class = {}

        for elem in self.data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[1], cut[2], cut[3])
            categors = self.data_catg[target_current]
            target_current_true = self.targeter[categors[category]]
            self.samples.append((path_current, target_current_true))
            if target_current_true not in self.samples_by_class:
                self.samples_by_class[target_current_true] = []
            self.samples_by_class[target_current_true].append(path_current)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target


def resize_and_save_images(dataset, output_root, size=(256, 256)):
    transform = transforms.Resize(size)
    # pdb.set_trace()
    for path, _ in tqdm(dataset.samples):
        # Create the new path
        rel_path = os.path.relpath(path, dataset.root)
        new_path = os.path.join(output_root, rel_path)
        # pdb.set_trace()
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(new_path), exist_ok=True)

        # Open, resize, and save the image
        with Image.open(path) as img:
            resized_img = transform(img)
            resized_img.save(new_path)


def create_dataset(data_dir, split, year, category, k=0):
    dataset = INatDataset(data_dir, split=split, year=year, category=category, k=k, transform=None)
    return dataset


# Example usage
if __name__ == "__main__":
    data_dir = "/datasets/inat_comp/2019/"
    output_dir = "/scratch/ssd004/scratch/kkasa/data/inat_comp/2019"
    split = "train"
    year = 2019
    category = "name"

    dataset = create_dataset(data_dir, split, year, category, )
    resize_and_save_images(dataset, output_dir)
