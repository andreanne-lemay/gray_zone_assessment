import os
import pandas as pd
import json
import math
import numpy as np
import torch
import matplotlib.image as mpimg
from monai.transforms import Compose
from sklearn.model_selection import train_test_split


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 data_path: str,
                 transforms: Compose,
                 label_colname: str = 'label',
                 image_colname: str = 'image'):
        self.df = df
        self.data_path = data_path
        self.transforms = transforms
        self.label_name = label_colname
        self.image_name = image_colname

    def __len__(self):
        return len(self.df)

    def __getitem__(self,
                    index: int):
        img_path = os.path.join(self.data_path, self.df[self.image_name].iloc[index])
        if img_path.endswith('.npy'):
            img = np.load(img_path).astype('float32')
        else:
            img = mpimg.imread(img_path).astype('float32')
            # Use provided bounding box if available. The bounding box coordinates should be stored in columns named
            # y1, y2, x1, x2.
            if 'y1' in self.df:
                idx_data = self.df.iloc[index]
                img = img[int(idx_data['y1']): int(idx_data['y2']), int(idx_data['x1']): int(idx_data['x2']), :]
                # Remove center crop if the bounding box is provided
                self.transforms = Compose([tr for tr in list(self.transforms.transforms)
                                           if 'CenterSpatialCrop' not in str(tr)])

        gt = self.df[self.label_name].iloc[index]
        # Image, label, image filename
        return self.transforms(img), \
               torch.as_tensor(int(gt)) if not math.isnan(gt) else gt, \
               self.df[self.image_name].iloc[index]


def loader(data_path: str,
           output_path: str,
           train_transforms: Compose,
           val_transforms: Compose,
           metadata_path: str = None,
           train_frac: float = 0.65,
           test_frac: float = 0.25,
           seed: int = 0,
           batch_size: int = 32,
           balanced: bool = False,
           label_colname: str = 'label',
           image_colname: str = 'image',
           split_colname: str = 'dataset',
           patient_colname: str = 'patient'):
    """
    Inspired by https://github.com/Project-MONAI/tutorials/blob/master/2d_classification/mednist_tutorial.ipynb

    Returns:
        DataLoader, DataLoader, DataLoader, pd.Dataframe: train dataset, validation dataset, val dataset, test dataset,
         test df
    """
    # Load metadata and create val/train/test split if not already done
    split_df = split_dataset(output_path, train_frac=train_frac, test_frac=test_frac,
                             seed=seed, metadata_path=metadata_path, split_colname=split_colname,
                             patient_colname=patient_colname)
    train_loader, val_loader, test_loader, weights = None, None, None, None
    df_train = split_df[split_df[split_colname] == "train"]
    if len(df_train):
        sampler, weights = get_balanced_sampler(df_train, label_colname)
        shuffle = not balanced
        train_ds = Dataset(df_train, data_path, train_transforms, label_colname, image_colname)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=shuffle, num_workers=10, sampler=sampler if balanced else None)

    df_val = split_df[split_df[split_colname] == "val"]
    if len(df_val):
        sampler, _ = get_balanced_sampler(df_val, label_colname)
        val_ds = Dataset(df_val, data_path, val_transforms, label_colname, image_colname)
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, num_workers=10, sampler=sampler if balanced else None)

    df_test = split_df[split_df[split_colname] == "test"]
    if len(df_test):
        test_ds = Dataset(df_test, data_path, val_transforms, label_colname, image_colname)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, num_workers=10)
    return train_loader, val_loader, test_loader, df_val, df_test, weights


def get_balanced_sampler(split_df: pd.DataFrame,
                         label_name: str):
    """ Balances the sampling of classes to have equal representation. """
    labels, count = np.unique(split_df[label_name], return_counts=True)
    weights = (1 / torch.Tensor(count)).float()
    sample_weights = torch.tensor([weights[int(l)] for l in split_df[label_name]]).float()
    sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(sample_weights))
    return sampler, weights


def split_dataset(output_path: str,
                  metadata_path: str,
                  train_frac: float,
                  test_frac: float,
                  seed: int,
                  split_colname: str,
                  patient_colname: str):
    """Load csv file containing metadata (image filenames, labels, patient ids, and val/train/test split)"""
    split_df_path = os.path.join(output_path, "split_df.csv")

    # If output_path / "split_df.csv" exists use the already split csv
    if os.path.isfile(split_df_path):
        df = pd.read_csv(split_df_path)
    # If output_path / "split_df.csv" doesn't exist: split images by patient using the train and test fractions
    else:
        df = pd.read_csv(metadata_path)
        # If images are not already split into val/train/test, split by patient
        if split_colname not in df:
            patient_lst = list(set(df['patient'].tolist()))
            train_patients, remain_patients = train_test_split(patient_lst, train_size=train_frac, random_state=seed)
            test_patients, val_patients = train_test_split(remain_patients, train_size=test_frac / (1 - train_frac),
                                                           random_state=seed)

            df[split_colname] = None
            df.loc[df[patient_colname].isin(train_patients), split_colname] = 'train'
            df.loc[df[patient_colname].isin(val_patients), split_colname] = 'val'
            df.loc[df[patient_colname].isin(test_patients), split_colname] = 'test'

        df.to_csv(split_df_path)

    return df
