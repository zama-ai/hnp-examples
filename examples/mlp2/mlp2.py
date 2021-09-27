from pathlib import Path

import hnumpy as hnp
import numpy
import torch
import torchvision
from hnumpy.config import CompilationConfig
from torchvision import datasets


def get_mnist_dataset(batch_size: int, num_workers: int = 0):
    test_data = datasets.MNIST(
        root=str(Path(__file__).resolve().absolute().parent.parent / "data"),
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    return torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)


MODEL_WEIGHTS_DIR = Path(__file__).parent / "weights"

FC1_WEIGHTS = MODEL_WEIGHTS_DIR / "fc1_weights.npy"
FC1_BIASES = MODEL_WEIGHTS_DIR / "fc1_biases.npy"

FC2_WEIGHTS = MODEL_WEIGHTS_DIR / "fc2_weights.npy"
FC2_BIASES = MODEL_WEIGHTS_DIR / "fc2_biases.npy"


class MLP2:

    def __init__(self, fc1_w, fc1_b, fc2_w, fc2_b) -> None:
        self.fc1_w = fc1_w
        self.fc1_b = fc1_b
        self.fc2_w = fc2_w
        self.fc2_b = fc2_b

    def forward(self, x):
        """
        input_data must be the flattened array for the MNIST sample
        """
        x = numpy.matmul(x, self.fc1_w)
        x = x + self.fc1_b
        x = numpy.maximum(x, 0)
        x = numpy.matmul(x, self.fc2_w)
        x = x + self.fc2_b
        return x

    __call__ = forward


def main():
    fc1_w = numpy.load(FC1_WEIGHTS).T
    fc1_b = numpy.load(FC1_BIASES).T
    fc2_w = numpy.load(FC2_WEIGHTS).T
    fc2_b = numpy.load(FC2_BIASES).T

    print(fc1_w.shape, fc1_b.shape, fc2_w.shape, fc2_b.shape)

    mnist_mlp2 = MLP2(fc1_w, fc1_b, fc2_w, fc2_b)

    batch_size = 20

    test_data_loader = get_mnist_dataset(batch_size)

    config = CompilationConfig(parameter_optimizer="genetic")

    data, target = iter(test_data_loader).next()
    data: numpy.ndarray = data.cpu().numpy()
    data = numpy.reshape(data, (data.shape[0], -1))
    target: numpy.ndarray = target.cpu().numpy()

    output = mnist_mlp2(data)
    clear_preds = numpy.argmax(output, 1)
    good_ones = list(map(lambda x: not x, clear_preds - target))
    print("Good ones", good_ones)
    print("Expected", clear_preds)
    print("Achieved", target)
    print("Accuracy", sum(good_ones) / len(good_ones))
    # preds = h(data)

    # compile the function
    fhe_mnist_mlp2_forward = hnp.compile_fhe(
        mnist_mlp2.forward,
        {
            "x": hnp.encrypted_ndarray(
                bounds=(0.0, 1.0),
                shape=(
                    batch_size,
                    28 * 28,
                ),
            )
        },
        config=config,
    )

    context = fhe_mnist_mlp2_forward.create_context()
    keys = context.keygen()

    fhe_outputs = fhe_mnist_mlp2_forward.encrypt_and_run(keys, data)
    fhe_preds = numpy.argmax(fhe_outputs, 1)
    good_ones = list(map(lambda x: not x, fhe_preds - target))
    print("Good ones", good_ones)
    print("Expected", fhe_preds)
    print("Achieved", target)
    print("Accuracy", sum(good_ones) / len(good_ones))

if __name__ == "__main__":
    main()
