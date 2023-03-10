from torch.utils.data import Dataset
from tests import _PATH_DATA
import pickle
import os
import torch
import pytest
import os.path

# dataset = MNIST(...)
# assert len(dataset) == N_train for training and N_test for test
# assert that each datapoint has shape [1,28,28] or [728] depending on how you choose to format
# assert that all labels are represented


class dataset(Dataset):
    def __init__(self, images, labels):
        self.data = images.view(-1, 1, 28, 28)
        self.labels = labels

    def __getitem__(self, item):
        return self.data[item].float(), self.labels[item]

    def __len__(self):
        return len(self.data)


def data_load(filepath):
    with open(filepath, "rb") as fb:
        images, labels = pickle.load(fb)
    return dataset(images, labels)


class TestClass:
    N_train = 40000
    N_test = 5000

    @pytest.mark.skipif(
        not os.path.exists(
            os.path.join(_PATH_DATA, "processed/corruptmnist_train.npz")
        ),
        reason="Data files not found",
    )
    def test_train(self):
        dataset = data_load(
            os.path.join(_PATH_DATA, "processed/corruptmnist_train.npz")
        )
        assert len(dataset.data) == self.N_train, "Data is incomplete"
        assert dataset.data.shape == (self.N_train, 1, 28, 28), "Data is of wrong shape"
        assert len(dataset.labels) == len(
            dataset.data
        ), "Number of labels does not correspond to number of data-points"
        assert torch.sum(torch.unique(dataset.labels)) == torch.tensor(
            45
        ), "Labels do not represent all classes."

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(_PATH_DATA, "processed/corruptmnist_test.npz")),
        reason="Data files not found",
    )
    def test_test(self):
        dataset = data_load(os.path.join(_PATH_DATA, "processed/corruptmnist_test.npz"))
        assert len(dataset.data) == self.N_test, "Data is incomplete"
        assert dataset.data.shape == (self.N_test, 1, 28, 28), "Data is of wrong shape"
        assert len(dataset.labels) == len(
            dataset.data
        ), "Number of labels does not correspond to number of data-points"
        assert torch.sum(torch.unique(dataset.labels)) == torch.tensor(
            45
        ), "Labels do not represent all classes."
