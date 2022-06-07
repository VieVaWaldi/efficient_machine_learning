from config import cfg

import random as rd

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class Fashion_mnist_data:
    """
        Task 3_2_1: Create training and test data from MNIST Fashion, then convert to tensor.
    """

    def __init__(self):

        self.DATA_ROOT = cfg["paths"]["data"]
        self.PDF_ROOT = cfg["paths"]["pdf"]
        self.BATCH_SIZE = cfg["paras"]["batch_size"]

        self.train_data = datasets.FashionMNIST(
            root=self.DATA_ROOT, train=True, download=True, transform=ToTensor())
        self.test_data = datasets.FashionMNIST(
            root=self.DATA_ROOT, train=False, download=True, transform=ToTensor())

        self.classes = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt",
                        "Sneaker", "Bag", "Ankle boot", ]

        self.train_dataloader, self.test_dataloader = self.wrap_data_in_dataloader()

        # Access data and label from data
        # print(self.test_data.data[0])
        # print(self.test_data.targets[0])

    def to_label(self, target):
        return self.classes[target]

    def wrap_data_in_dataloader(self):
        """
            Task 3_2_3: Wrap the data in dataloaders, use batch_size.
        """

        train_dataloader = DataLoader(self.train_data, self.BATCH_SIZE)
        test_dataloader = DataLoader(self.test_data, self.BATCH_SIZE)

        print(f"Original train_data length {len(self.train_data)}")
        print(f"Dataloder train_data length {len(train_dataloader)}. ",
              f"Times batch_size = {len(train_dataloader)*self.BATCH_SIZE}. Almost the same :)")

        return train_dataloader, test_dataloader

    def save_random_image(self):
        data_idx = rd.randint(0, len(self.train_data.data))

        plt.title(self.to_label(self.train_data.targets[data_idx]))
        print(self.train_data.data[data_idx].shape)
        plt.imshow(self.train_data.data[data_idx])
        self.save_current_fig(
            f"{self.to_label(self.train_data.targets[data_idx])}_#{data_idx}")

    def save_current_fig(self, name):
        """
            Task 3_2_2: Visualize some images with labels.
        """
        path = f"{self.PDF_ROOT}{name}.pdf"
        with PdfPages(path) as pdf:
            pdf.savefig()
            plt.close()
        print(f"Saved random pdf succesfully at {path}.")
