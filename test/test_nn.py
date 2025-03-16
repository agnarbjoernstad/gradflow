from gradflow.nn.modules import container, linear, activation
from gradflow.nn.modules.loss import CrossEntropyLoss
from gradflow.optim.adam import Adam
from script.mnist import read_labels, read_images, run_epoch
import numpy as np


def test_train_mnist():
    np.random.seed(42)

    BATCH_SIZE = 128
    EPOCHS = 2
    LR = 0.001

    model = container.Sequential(
        linear.Linear(784, 128),
        activation.ReLU(),
        linear.Linear(128, 128),
        activation.ReLU(),
        linear.Linear(128, 10),
        activation.Softmax(),
    )
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)

    mnist_path = "data/mnist"
    train_labels_path = f"{mnist_path}/train-labels-idx1-ubyte.gz"
    train_x_path = f"{mnist_path}/train-images-idx3-ubyte.gz"
    val_labels_path = f"{mnist_path}/t10k-labels-idx1-ubyte.gz"
    val_x_path = f"{mnist_path}/t10k-images-idx3-ubyte.gz"

    train_x = read_images(train_x_path)
    train_y = read_labels(train_labels_path)
    val_x = read_images(val_x_path)
    val_y = read_labels(val_labels_path)

    for epoch in range(EPOCHS):
        _, train_correct, num_train_samples = run_epoch(
            model,
            train_x,
            train_y,
            loss_fn,
            BATCH_SIZE,
            epoch,
            EPOCHS,
            optimizer,
            train=True,
        )
        _, val_correct, num_val_samples = run_epoch(
            model,
            val_x,
            val_y,
            loss_fn,
            BATCH_SIZE,
            epoch,
            EPOCHS,
            optimizer,
            train=False,
        )

    train_accuracy = train_correct / num_train_samples
    val_accuracy = val_correct / num_val_samples

    assert train_accuracy > 0.9
    assert train_accuracy < 0.99
    assert val_accuracy > 0.9
    assert val_accuracy < 0.99
