import os
import torch
from typing import Union, List, Tuple
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import Resize, Normalize, Compose
from sklearn.model_selection import train_test_split


class ClsDataset(Dataset):
    def __init__(self, df, img_dir):
        self.df = df
        self.img_dir = img_dir
        self.transforms = Compose([
            Resize((300, 300)),
            Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        image = read_image(os.path.join(self.img_dir, img_name)).float()
        label = torch.Tensor(self.df.iloc[idx, 1:])

        image = self.transforms(image)

        return image, label


def split_define_dataloader(dataframe, img_dir, batch_size=8, train_size=0.8, val_size=0.1, test_size=0.1,
                            shuffle=True):
    if train_size + val_size + test_size != 1.:
        raise ValueError('train_size + val_size + test_size is not equal 1')

    df_train, df_test = train_test_split(dataframe, train_size=train_size, shuffle=shuffle)
    train_dataset = ClsDataset(df_train, img_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    yield train_dataloader

    if test_size != 0. and val_size != 0:
        df_val, df_test = train_test_split(df_test, train_size=val_size / (val_size + test_size))
        val_dataset = ClsDataset(df_val, img_dir)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        yield val_dataloader

    test_dataset = ClsDataset(df_test, img_dir)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    yield test_dataloader


class ClsDatasetMultiModel(Dataset):
    def __init__(self, df, img_dir, num_models, num_cls_per_model):
        self.df = df
        self.img_dir = img_dir
        self.num_models = num_models
        if type(num_cls_per_model) is int:
            num_cls_per_model = [num_cls_per_model] * num_models
        self.num_cls_model = num_cls_per_model
        self.transforms = Compose([
            Resize((300, 300)),
            Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        image = read_image(os.path.join(self.img_dir, img_name)).float()

        start_col = 1
        labels = []
        sample_weight = []
        for model_idx in range(self.num_models):
            labels.append(torch.Tensor(self.df.iloc[idx, start_col:start_col + self.num_cls_model[model_idx]]))
            sample_weight.append(torch.Tensor([float(labels[model_idx].isnan().sum() == 0)]))

            labels[model_idx] = labels[model_idx].nan_to_num(0.5)
            start_col += self.num_cls_model[model_idx]

        image = self.transforms(image)

        return image, labels, sample_weight


def split_define_dataloader_mm(dataframe, img_dir, number_models, num_cls_per_model,
                               batch_size=8, train_size=0.8, val_size=0.1, test_size=0.1, shuffle=True):
    if train_size + val_size + test_size != 1.:
        raise ValueError('train_size + val_size + test_size is not equal 1')

    df_train, df_test = train_test_split(dataframe, train_size=train_size, shuffle=shuffle)
    train_dataset = ClsDatasetMultiModel(df_train, img_dir, number_models, num_cls_per_model)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    yield train_dataloader

    if test_size != 0. and val_size != 0:
        df_val, df_test = train_test_split(df_test, train_size=val_size / (val_size + test_size))
        val_dataset = ClsDatasetMultiModel(df_val, img_dir, number_models, num_cls_per_model)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        yield val_dataloader

    test_dataset = ClsDatasetMultiModel(df_test, img_dir, number_models, num_cls_per_model)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    yield test_dataloader


def data_loader_generator(dataframes: Union[List, Tuple], img_dirs: Union[List, Tuple], num_models: int,
                          num_cls_per_model: Union[List, int], batch_size: int = 8):
    for index, (df, img_dir) in enumerate(zip(dataframes, img_dirs)):
        dataset = ClsDatasetMultiModel(df, img_dir, num_models, num_cls_per_model)
        yield DataLoader(dataset, batch_size=batch_size, shuffle=(index == 0))


if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    df_path = r'D:\Machine Learning Project\5kCompliance\dataset\train\train_meta.csv'
    df = pd.read_csv(df_path, usecols=['fname', 'mask', 'distancing', '5k'])

    # dataloader_gen = split_define_dataloader(dataframe=df,
    #                                          train_size=0.9, val_size=0.1, test_size=0.0, batch_size=8,
    #                                          img_dir=r'D:\Machine Learning Project\5kCompliance\dataset\train\images',)
    dataloader_gen = split_define_dataloader_mm(dataframe=df, number_models=2, num_cls_per_model=[1, 1],
                                                train_size=0.9, val_size=0.1, test_size=0.0, batch_size=8,
                                                img_dir=r'D:\Machine Learning Project\5kCompliance\dataset\train\images', )
    dataloader = {'train': next(dataloader_gen), 'val': next(dataloader_gen), 'test': next(dataloader_gen, None)}

    print(len(dataloader['train'].dataset))
    dataset_sizes = {key: (len(value.dataset) if value is not None else None) for key, value in dataloader.items()}
    print(dataset_sizes)

    train_image, train_label, train_sample_weight = next(iter(dataloader['train']))
    print(train_image.size())
    # print(train_label.size())

    img = train_image[4].squeeze().numpy()
    img = np.moveaxis(img, 0, -1)
    plt.imshow(img)
    plt.show()

    label = train_label
    print(label[1])
    print(train_sample_weight[1])
    print(label[1] * train_sample_weight[1])
