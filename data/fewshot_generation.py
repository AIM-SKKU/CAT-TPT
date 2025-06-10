import math
import os

import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import PIL
from PIL import Image
import random


class BaseJsonDataset(Dataset):
    def __init__(self, image_path, json_path, encode_image, mode='train',  n_shot=None, transform=None, device=None):
        self.device =device
        self.encode_image = encode_image
        self.transform = transform
        self.image_path = image_path
        self.split_json = json_path
        self.mode = mode
        self.image_list = []
        self.label_list = []
        self.tform = transforms.Compose([transforms.ToTensor(), transforms.Resize((500, 500), ), ])
        with open(self.split_json) as fp:
            splits = json.load(fp)
            samples = splits[self.mode]
            for s in samples:
                self.image_list.append(s[0]) #### whole dataset
                self.label_list.append(s[1])

        # #### random sample n_shot image for each class
        if n_shot is not None:
            few_shot_samples = []
            c_range = max(self.label_list) + 1
            for c in range(c_range):
                c_idx = [idx for idx, lable in enumerate(self.label_list) if lable == c]
                random.seed(0)
                few_shot_samples.extend(random.sample(c_idx, n_shot))
            self.image_list = [self.image_list[i] for i in few_shot_samples]
            self.label_list = [self.label_list[i] for i in few_shot_samples]
        else:
        #### random sample 1k image
            num_samples = 1000
            all_indices = list(range(len(self.image_list)))
            print(len(self.image_list))
            sampled_indices = random.sample(all_indices, num_samples)
            self.image_list = [self.image_list[i] for i in sampled_indices]
            self.label_list = [self.label_list[i] for i in sampled_indices]


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_path, self.image_list[idx])
        image = self.transform(Image.open(image_path).convert('RGB')).unsqueeze(0).to(self.device)
        image_embeddings = self.encode_image(image)

        init_image = self.tform(Image.open(image_path).convert('RGB'))
        return self.image_list[idx], image_embeddings, init_image


fewshot_datasets = ['DTD', 'Flower102', 'Food101', 'Cars', 'SUN397',
                    'Aircraft', 'Pets', 'Caltech101', 'UCF101', 'eurosat']

path_dict = {
    # dataset_name: ["image_dir", "json_split_file"]
    "flower102": ["jpg", "split_zhou_OxfordFlowers.json"],
    "food101": ["images", "split_zhou_Food101.json"],
    "dtd": ["images", "split_zhou_DescribableTextures.json"],
    "pets": ["images", "split_zhou_OxfordPets.json"],
    "sun397": ["SUN397", "split_zhou_SUN397.json"],
    "caltech101": ["101_ObjectCategories", "split_zhou_Caltech101.json"],
    "ucf101": ["UCF-101-midframes", "split_zhou_UCF101.json"],
    "cars": ["", "split_zhou_StanfordCars.json"],
    "eurosat": ["2750", "split_zhou_EuroSAT.json"]
}

ID_to_DIRNAME={
    'I': 'imageNet',
    'A': 'imagenet-a',
    'K': 'ImageNet-Sketch',
    'R': 'imagenet-r',
    'V': 'imagenetv2-matched-frequency-format-val',
    'flower102': 'oxford_flowers',
    'dtd': 'dtd',
    'pets': 'oxford_pets',
    'cars': 'stanford_cars',
    'ucf101': 'ucf101',
    'caltech101': 'caltech-101',
    'food101': 'food-101',
    'sun397': 'sun397',
    'aircraft': 'fgvc_aircraft',
    'eurosat': 'eurosat',
    'C': 'imagenet-c'
}
def build_fewshot_dataset(set_id, root, transform, encode_image, mode='train', n_shot=None, device=None):
    if set_id.lower() == 'aircraft':
        return Aircraft(root, encode_image, mode, n_shot, transform, device)
    path_suffix, json = path_dict[set_id.lower()]
    image_path = os.path.join(root, path_suffix)
    json_path = os.path.join(root, json)
    return BaseJsonDataset(image_path, json_path, encode_image, mode, n_shot, transform, device)


class Aircraft(Dataset):
    """ FGVC Aircraft dataset """

    def __init__(self, root, encode_image, mode='train', n_shot=None, transform=None, device=None):
        self.device =device
        self.encode_image = encode_image
        self.transform = transform
        self.tform = transforms.Compose([transforms.ToTensor(), transforms.Resize((500, 500), ), ])

        self.transform = transform
        self.path = root
        self.mode = mode

        self.cname = []
        with open(os.path.join(self.path, "variants.txt"), 'r') as fp:
            self.cname = [l.replace("\n", "") for l in fp.readlines()]

        self.image_list = []
        self.label_list = []
        with open(os.path.join(self.path, 'images_variant_{:s}.txt'.format(self.mode)), 'r') as fp:
            lines = [s.replace("\n", "") for s in fp.readlines()]
            for l in lines:
                ls = l.split(" ")
                img = ls[0]
                label = " ".join(ls[1:])
                self.image_list.append("{}.jpg".format(img))
                self.label_list.append(self.cname.index(label))

        if n_shot is not None:
            few_shot_samples = []
            c_range = max(self.label_list) + 1
            for c in range(c_range):
                c_idx = [idx for idx, lable in enumerate(self.label_list) if lable == c]
                random.seed(0)
                few_shot_samples.extend(random.sample(c_idx, n_shot))
            self.image_list = [self.image_list[i] for i in few_shot_samples]
            self.label_list = [self.label_list[i] for i in few_shot_samples]

        else:
        #### random sample 1k image
            num_samples = 1000
            all_indices = list(range(len(self.image_list)))
            print(len(self.image_list))
            sampled_indices = random.sample(all_indices, num_samples)
            self.image_list = [self.image_list[i] for i in sampled_indices]
            self.label_list = [self.label_list[i] for i in sampled_indices]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.path, 'images', self.image_list[idx])
        image = self.transform(Image.open(image_path).convert('RGB')).unsqueeze(0).to(self.device)
        image_embeddings = self.encode_image(image)

        init_image = self.tform(Image.open(image_path).convert('RGB'))
        return self.image_list[idx], image_embeddings, init_image


class CustomTextDataset(Dataset):
    def __init__(self, labels, lable_img_dict, k=50, is_train=False, test_freq=5):
        self.labels = labels
        self.lable_img_dict = lable_img_dict
        self.k = k


        for label in self.labels:
            img_id = self.lable_img_dict[label]
            img_id_unique = np.unique(img_id)
            img_id_unique_test = img_id_unique[::test_freq] # split, use every test_freq'th sample for test
            img_id_unique_train = [i for i in img_id_unique if i not in img_id_unique_test]
            #
            # #split images by img id

            if is_train:
                self.lable_img_dict[label] = [lable_img_dict[label][img_id.index(i)] for i in img_id_unique_train]
            else:
                self.lable_img_dict[label] = [lable_img_dict[label][img_id.index(i)] for i in img_id_unique_test]


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label_name = self.labels[idx]
        img_id = self.lable_img_dict[label_name]
        chosen_inds = np.random.choice(np.arange(len(img_id)), self.k)
        chosen_img_id = [img_id[i] for i in chosen_inds]

        return label_name, chosen_img_id
        # return self.labels, chosen_img_id