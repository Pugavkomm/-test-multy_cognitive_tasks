import torch

from cgtasknet.instruments.instrument_pca import PCA


def test_pca_number_of_components_shape():
    data = torch.zeros((10, 10))
    pca1 = PCA(1)
    pca2 = PCA(2)
    pca3 = PCA(3)
    pca4 = PCA(4)

    pc1 = pca1.decompose(data)
    pc2 = pca2.decompose(data)
    pc3 = pca3.decompose(data)
    pc4 = pca4.decompose(data)

    assert pc1.shape == (10, 1)
    assert pc2.shape == (10, 2)
    assert pc3.shape == (10, 3)
    assert pc4.shape == (10, 4)


def test_pca1_calculate_zero_data():
    data = torch.zeros((10, 10))
    test_data = torch.zeros((10, 1))
    pca1 = PCA(1)
    pc1 = pca1.decompose(data)
    for i in range(10):
        assert test_data[i, 0] == pc1[i, 0]


def test_pca2_calculate_zero_data():
    data = torch.zeros((10, 10))
    test_data = torch.zeros((10, 2))
    pca2 = PCA(2)
    pc2 = pca2.decompose(data)
    for i in range(10):
        for j in range(2):
            assert test_data[i, j] == pc2[i, j]


# TODO: 1. test cetnring data; 2. test random data; 3. test derivation!!!
