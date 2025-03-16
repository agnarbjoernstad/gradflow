from gradflow.nn.modules import container, linear, activation
from gradflow.nn.modules.loss import CrossEntropyLoss, MSELoss
from gradflow.optim.adam import Adam
from gradflow import tensor
from script.mnist import read_labels, read_images, run_epoch
from tqdm import tqdm
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


def test_train_sin():
    np.random.seed(42)

    BATCH_SIZE = 128
    EPOCHS = 50
    LR = 1e-3
    TRAIN_SAMPLES = 2000
    VAL_SAMPLES = 200

    train_x_min = -np.pi
    train_x_max = np.pi
    val_x_min = -np.pi
    val_x_max = np.pi

    model = container.Sequential(
        linear.Linear(1, 128),
        activation.ReLU(),
        linear.Linear(128, 1),
    )
    loss_fn = MSELoss()
    optimizer = Adam(model.parameters(), lr=LR)

    train_x = np.random.uniform(train_x_min, train_x_max, TRAIN_SAMPLES)
    train_y = np.sin(train_x)
    val_x = np.random.uniform(val_x_min, val_x_max, VAL_SAMPLES)
    val_y = np.sin(val_x)

    for epoch in range(EPOCHS):
        train_loss = 0
        train_bar = tqdm(
            range(0, len(train_x), BATCH_SIZE),
            desc="{dataset_str} epoch {epoch}/{epochs}",
        )
        for start_idx in train_bar:
            end_idx = min(start_idx + BATCH_SIZE, len(train_x))
            x = tensor(train_x[start_idx:end_idx]).unsqueeze(-1)
            y = tensor(train_y[start_idx:end_idx]).unsqueeze(-1)
            p = model(x)
            loss = loss_fn(p, y)
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.detach().numpy()[0] * (end_idx - start_idx)
            optimizer.step()
            train_bar.set_description(
                f"Train epoch {epoch}/{EPOCHS}: mse loss {(train_loss/end_idx):.5f}"
            )
        train_loss /= len(train_x)

        val_loss = 0
        val_bar = tqdm(
            range(0, len(val_x), BATCH_SIZE),
            desc="Val epoch {epoch}/{epochs}",
        )
        for start_idx in val_bar:
            end_idx = min(start_idx + BATCH_SIZE, len(val_x))
            x = tensor(val_x[start_idx:end_idx]).unsqueeze(-1)
            y = tensor(val_y[start_idx:end_idx]).unsqueeze(-1)
            p = model(x)
            loss = loss_fn(p, y)
            val_loss += loss.detach().numpy()[0] * (end_idx - start_idx)
            val_bar.set_description(
                f"Val epoch {epoch}/{EPOCHS}: mse loss {(val_loss/end_idx):.5f}"
            )
        val_loss /= len(val_x)

    assert train_loss < 0.01
    assert train_loss > 0
    assert val_loss < 0.01
    assert val_loss > 0
