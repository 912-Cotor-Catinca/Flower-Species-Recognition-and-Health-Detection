import os
import cv2
from torch.utils.data import Dataset
from tqdm import tqdm


class CustomDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        target = self.targets[index]

        if self.transform:
            sample = self.transform(sample)
        return sample, target


dataset_dir = "C:/Users/Catinca/Documents/TomatoDataset/"


def load_dataset(data_dir):
    # Load images and extract labels from directory names
    max_nr_samples = 1816
    count = 0
    sclass = 0
    for label in os.listdir(dataset_dir):
        label_dir = os.path.join(dataset_dir, label)
        for img in tqdm(os.listdir(label_dir)):
            ext = os.path.splitext(img)[-1].lower()
            if ext == ".jpg" and count < max_nr_samples:
                count += 1
                image_path = os.path.join(label_dir, img)
                image = cv2.imread(image_path)
                data_dir.append([image, sclass])
            elif ext != ".jpg" and count < max_nr_samples:
                print(count)
                new_path = label_dir + "/" + img
                for file in os.listdir(new_path):
                    if count < max_nr_samples:
                        count += 1
                        image_path = os.path.join(new_path, file)
                        image = cv2.imread(image_path)
                        data_dir.append([image, sclass])
        print(label, sclass)
        print(count)
        count = 0
        sclass += 1