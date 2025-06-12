from api.MO810_API import MO810Dataset, MO810DataModule

from torch.utils.data import Dataset

import os
import glob
import h5py

from PIL import Image

import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset, random_split
from torchvision.transforms.v2 import Compose, ToImage, ToDtype, Normalize, Resize

class PCamDataset(MO810Dataset):
    """
    Custom PyTorch Dataset for loading the Patch Camelyon (PCam) image classification dataset.
    """

    def __init__(self, data_dir, download=True, transform=None):
        """
        Initializes the PCamDataset.
        Args:
            data_dir (str): Directory to store or load the dataset from.
            download (bool): Whether to download the dataset if not found.
            transform (callable, optional): Transformations to apply to each image.
        """
        super().__init__()
        # Data dir
        self.data_dir = data_dir
        self.dataset_dir = self.data_dir + "/pcam"
        self.transform = transform

        if download:
            self.remote_url = "https://www.kaggle.com/api/v1/datasets/download/andrewmvd/metastatic-tissue-classification-patchcamelyon"
            self.local_filename = "pcam.zip"
            self.download_data()

        sets = self.__read_hdf5([("training", "train"), ("validation", "valid"), ("test", "test")])
        (self.train_set, self.val_set, self.test_set) = sets
        self.full_set = ConcatDataset(sets)
        self.labels = []

        # Count samples per class
        self.classes = ["healthy", "tumor"]
        self.nsamples = { 0: 0, 1: 0 }
        for dataset in sets:
            for l in dataset.labels:
                self.labels.append(l)
                self.nsamples[l] += 1

    def download_data(self):
        """
        Downloads and unzips the dataset if not already available locally.
        """
        if not os.path.exists(self.local_filename):
            # Download the dataset zip file 
            import urllib.request
            print(f"Downloading {self.local_filename} from {self.remote_url}")
            urllib.request.urlretrieve(self.remote_url, self.local_filename)

        if not os.path.exists(self.data_dir):
            print(f"Creating the data folder")
            os.mkdir(self.data_dir)

        if not os.path.exists(self.dataset_dir):
            # Unzip the dataset file
            print(f"{self.dataset_dir}")
            import zipfile
            print(f"Extracting {self.local_filename} into {self.data_dir}")
            with zipfile.ZipFile(self.local_filename, 'r') as zip_ref:
                zip_ref.extractall(path=self.data_dir)

    def __len__(self):
        """
        Returns:
            int: Total number of samples in the dataset.
        """
        return len(self.full_set)
    
    def convert_img(self, img, bkg_color=(255,255,255)):
        """
        Converts image to RGB and removes transparency if present.
        Args:
            img (PIL.Image): Input image.
            bkg_color (tuple): RGB color to use as background for transparency.
        Returns:
            PIL.Image: RGB image with no alpha channel.
        """
        if img.mode == 'P':
            if 'transparency' in img.info:
                img = img.convert('RGBA')
            else:
                img = img.convert('RGB')
        if img.mode == 'RGBA':
            # Remove alpha by compositing over background
            background = Image.new("RGB", img.size, bkg_color)
            background.paste(img, mask=img.split()[3])  # 3 is the alpha channel
            return background
        return img.convert('RGB')

    def __getitem__(self, index):
        """
        Fetches a single image-label pair by index.
        Args:
            index (int): Index of the sample.
        Returns:
            tuple: (image, label), where image is a transformed tensor and label is an integer.
        """
        image, label = self.full_set[index]
        image = self.convert_img(Image.fromarray(image))
        if self.transform:
            image = self.transform(image)
        return image, label
    
    def __str__(self):
        """
        Returns:
            str: Human-readable description of the dataset.
        """
        return f"PatchCamelyon Dataset (# samples = {self.nsamples})"
    
    def __read_hdf5(self, split_names: list[tuple[str, str]]):
        return [H5Dataset(self.data_dir, f"pcam/{x_name}_split", f"Labels/Labels/camelyonpatch_level_2_split_{y_name}_y") for (x_name, y_name) in split_names]

class H5Dataset(Dataset):
    def __init__(self, data_dir, data_path, labels_path, transform=None):
        self.data_file = h5py.File(f"{data_dir}/{data_path}.h5", "r")
        self.labels_file = h5py.File(f"{data_dir}/{labels_path}.h5", "r")
        
        self.data = self.data_file["x"]
        self.labels = [l.squeeze()[()] for l in self.labels_file["y"]]

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y

    def __del__(self):
        self.data_file.close()
        self.labels_file.close()

class TransformedSubset(Dataset):
    """
    A dataset wrapper for applying a different transform to a subset of a dataset.

    This is useful when you want to change the transformation pipeline (e.g., data augmentation)
    for a specific subset, such as using a different transform for validation or testing
    while keeping the original dataset unchanged.

    Attributes:
        subset (torch.utils.data.Subset): The original subset of the dataset.
        transform (callable, optional): A function/transform that takes in a data sample and returns a transformed version.
    """

    def __init__(self, subset, transform=None):
        """
        Initialize the TransformedSubset.
        Args:
            subset (torch.utils.data.Subset): The subset to wrap.
            transform (callable, optional): Transform to apply to the input data.
        """
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        """
        Retrieve an item from the subset and apply the transform to the data.
        Args:
            idx (int): Index of the data sample to retrieve.
        Returns:
            tuple: (transformed_data, target), where target is the label or ground truth.
        """
        data, label = self.subset[idx]
        if self.transform:
            data = self.transform(data)
        return data, label

    def __len__(self):
        """
        Get the number of samples in the subset.
        Returns:
            int: Length of the dataset.
        """
        return len(self.subset)

class PCamDataModule(MO810DataModule):
    """
    PyTorch Lightning DataModule for the PCamDataset.
    Handles data loading, splitting into train/val/test subsets,
    and setting up data loaders.
    """

    def __init__(self, data_dir: str = "./data/", 
                 train_transform = None,
                 val_transform = None,
                 test_transform = None,
                 batch_size: int = 32, 
                 num_workers: int = 4):
        """
        Initializes the PCamDataModule.

        Default transform pipeline: ToImage() => reize((128,128)) => ToDtype(float32, scale=True) => Normalize())

        Args:
            data_dir (str): Directory where the dataset is stored or downloaded.
            train_transform: Transform pipeline to be applied to the train set. If None, the default pipeline is applied.
            val_transform: Transform pipeline to be applied to the validation set. If None, the default pipeline is applied.
            test_transform: Transform pipeline to be applied to the test set. If None, the default pipeline is applied.
            batch_size (int): Batch size for the data loaders.
            num_workers (int): Number of subprocesses for data loading.
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Precomputed normalization stats (post-transforms)
        # Transforms: ToImage(), Resize((128,128)), and ToDtype(scale=True)
        self.precomputed_dataset_stats = {'mean': torch.tensor([0.7268, 0.7133, 0.7056]), 
                                          'std': torch.tensor([0.3064, 0.3112, 0.3173])}

        # Validation and test set image transformation pipeline
        self.default_transform_pipeline = Compose([ToImage(), 
                                                   Resize((128, 128)),
                                                   ToDtype(torch.float32, scale=True),
                                                   Normalize(self.precomputed_dataset_stats["mean"],
                                                             self.precomputed_dataset_stats["std"])])

        # Load full dataset with transforms
        self.full_dataset = PCamDataset(data_dir=self.data_dir)

        # Split into training, validation, and test subsets
        self.train_subset, self.val_subset, self.test_subset = self.full_dataset.train_set, self.full_dataset.val_set, self.full_dataset.test_set

        # Set the transform pipelines
        if train_transform:
            self.train_transform = train_transform
        else:
            self.train_transform = self.default_transform_pipeline
        if val_transform:
            self.val_transform = val_transform
        else:
            self.val_transform = self.default_transform_pipeline
        if test_transform:
            self.test_transform = test_transform
        else:
            self.test_transform = self.default_transform_pipeline

    def sample_dataset(self, dataset, fraction=None, samples_per_class=None):
        """
        Returns a sampled subset of the dataset based on a fraction of the total dataset
        or a fixed number of samples per class.
        Args:
            dataset (Dataset): The dataset to sample from.
            fraction (float, optional): Fraction of the dataset to use (0 < fraction ≤ 1).
            samples_per_class (int, optional): Number of samples per class to include.

        Returns:
            Subset: A subset of the dataset with the selected samples.

        Raises:
            ValueError: If both `fraction` and `samples_per_class` are provided.
        """
        if fraction != None and samples_per_class != None:
            raise ValueError("SneakersDataModule ERROR: Cannot sample dataset using both fraction and samples_per_class")
        elif fraction:
            N = int(len(dataset) * fraction)
            return Subset(dataset, range(N))
        elif samples_per_class:
            count = [0] * len(self.full_dataset.classes)
            indices = []
            for i, (_, label) in enumerate(dataset):
                if count[label] < samples_per_class: 
                    count[label] += 1
                    indices.append(i)
            return Subset(dataset, indices)
        else:
            return dataset

    def train_dataloader(self, fraction=None, samples_per_class=None, transform=None):
        """
        Returns a dataloader for the training set.
        Args:
            fraction (float, optional): Fraction of the training dataset to use (0 < fraction ≤ 1).
            samples_per_class (int, optional): Number of samples per class to include.
            transform: If != None override the transform defined at init.

        Returns:
            DataLoader: DataLoader for training set.
        """
        if not transform: transform = self.train_transform
        train_subset = self.sample_dataset(self.train_subset, fraction, samples_per_class)
        return DataLoader(dataset=TransformedSubset(subset=train_subset, 
                                                    transform=transform),
                          batch_size=self.batch_size, 
                          shuffle=True, 
                          num_workers=self.num_workers)

    def val_dataloader(self, transform=None):
        """
        Args:
            transform: If != None override the transform defined at init.
        Returns:
            DataLoader: DataLoader for validation set.
        """
        if not transform: transform = self.val_transform
        return DataLoader(dataset=TransformedSubset(subset=self.val_subset, 
                                                    transform=transform),
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers)

    def test_dataloader(self, transform=None):
        """
        Args:
            transform: If != None override the transform defined at init.
        Returns:
            DataLoader: DataLoader for test set.
        """
        if not transform: transform = self.test_transform
        return DataLoader(dataset=TransformedSubset(subset=self.test_subset, 
                                                    transform=transform),
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers)
    