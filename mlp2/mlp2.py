from pathlib import Path

import numpy
import hnumpy as hnp

import torch
import torchvision
from torchvision import datasets

from hnumpy.config import CompilationConfig


def get_mnist_dataset(batch_size: int, num_workers: int = 0):
    test_data = datasets.MNIST(
        root="data/",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    return torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

fc1_w = None
fc1_b = None
fc2_w = None
fc2_b = None

def mnist_mlp2_forward(x: numpy.ndarray):
    """
    input_data must be the flattened array for the MNIST sample
    """
    x = numpy.matmul(x, fc1_w)
    x = x + fc1_b
    x = numpy.maximum(x, 0)
    x = numpy.matmul(x, fc2_w)
    x = x + fc2_b
    return x

MODEL_WEIGHTS_DIR = Path(__file__).parent / "weights"

FC1_WEIGHTS = MODEL_WEIGHTS_DIR / "fc1_weights.npy"
FC1_BIASES = MODEL_WEIGHTS_DIR / "fc1_biases.npy"

FC2_WEIGHTS = MODEL_WEIGHTS_DIR / "fc2_weights.npy"
FC2_BIASES = MODEL_WEIGHTS_DIR / "fc2_biases.npy"

def main():
    global fc1_w, fc1_b, fc2_w, fc2_b
    fc1_w = numpy.load(FC1_WEIGHTS).T
    fc1_b = numpy.load(FC1_BIASES).T
    fc2_w = numpy.load(FC2_WEIGHTS).T
    fc2_b = numpy.load(FC2_BIASES).T

    print(fc1_w.shape, fc1_b.shape, fc2_w.shape, fc2_b.shape)

    batch_size = 2

    test_data_loader = get_mnist_dataset(batch_size)

    config = CompilationConfig(parameter_optimizer="handselected")

    data, target = iter(test_data_loader).next()
    data: numpy.ndarray = data.cpu().numpy()
    data = numpy.reshape(data, (data.shape[0], -1))
    target: numpy.ndarray = target.cpu().numpy()

    output = mnist_mlp2_forward(data)
    clear_preds = numpy.argmax(output, 1)
    print(list(map(lambda x: not x, clear_preds - target)))
    print(clear_preds)
    print(target)
    # preds = h(data)

    # compile the function
    fhe_mnist_mlp2_forward = hnp.homomorphic_fn(
        mnist_mlp2_forward,
        hnp.encrypted_ndarray(bounds=(0.0, 1.0), shape=(batch_size, 28 * 28,)),
        config=config,
    )

    fhe_outputs = fhe_mnist_mlp2_forward(data)[0]
    fhe_preds = numpy.argmax(fhe_outputs, 1)
    print(list(map(lambda x: not x, fhe_preds - target)))
    print(fhe_preds)
    print(target)


if __name__ == "__main__":
    main()
