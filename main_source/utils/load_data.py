import numpy as np
import os
import torch
import csv
from PIL import Image, ImageOps
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


def load_data_from_csv(dict_config):
    train_dataset, validate_dataset = DatasetReader(dict_config['link_train_csv'], dict_config['root_folder_train'],
                                                    size = (dict_config['size_width'], dict_config['size_height'])), \
                                      DatasetReader(dict_config['link_val_csv'], dict_config['root_folder_val'],
                                                    size=(dict_config['size_width'], dict_config['size_height']))

    train_loader, validate_loader = DataLoader(train_dataset, batch_size=dict_config['batch_size'], shuffle=True,
                                               num_workers=2, pin_memory=True), \
                                    DataLoader(validate_dataset, batch_size=dict_config['batch_size'], shuffle=False,
                                               num_workers=2, pin_memory=True)
    return train_loader, validate_loader


def _default_flist_reader(path):
    """
    flist format:
    """
    list_data = []
    with open(path) as csvDataFile:
        csvReader = csv.reader(csvDataFile, delimiter=',')
        for row in csvReader:
            image_name = row[0]
            class_label = int(row[5]) - 1
            box_cordinate = (int(row[1]), int(row[2]), int(row[3]), int(row[4]))
            list_data.append((image_name, class_label, box_cordinate))
    return list_data


class DatasetReader(torch.utils.data.Dataset):
    def __init__(self, path_to_csv, root_path, size = None, f_reader=_default_flist_reader):
        self.root_path = root_path
        self.data_version_path = path_to_csv
        self.data_list = f_reader(self.data_version_path)
        # transform
        self.transform = transforms.Compose([
            transforms.Resize(size=(size[0], size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, item):
        image = Image.open(os.path.join(self.root_path, self.data_list[item][0])).convert('RGB')  # (C, H, W)
        image = self.transform(image)
        # filename of image should have 'id_label.jpg/png' form
        label = self.data_list[item][1]
        return image, label

    def __len__(self):
        return len(self.data_list)
