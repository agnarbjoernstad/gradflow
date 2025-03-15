from gradflow.nn.modules import linear, module, container, activation
from gradflow.nn.modules.loss import MSELoss, CrossEntropyLoss
from gradflow.autograd.grad_mode import no_grad, set_grad_enabled
from gradflow.nn.functional import relu, softmax
from gradflow.optim.adam import Adam
from gradflow.optim.optimizer import Optimizer
import gzip
import matplotlib.pyplot as plt
import numpy as np
import gradflow as gf
import cv2
from typing import Optional
from tqdm import tqdm


def read_labels(path):
    with gzip.open(path, "r") as f:
        f.read(8)
        labels = []
        while True:
            buf = f.read(1)
            if not buf:
                break
            labels.append(
                gf.tensor(np.frombuffer(buf, dtype=np.uint8).astype(np.int64))
            )
    return gf.tensor(labels)


def read_images(path):
    image_size = 28
    images = []
    with gzip.open(path, "r") as f:
        f.read(16)
        while True:
            buf = f.read(image_size * image_size)
            if not buf:
                break
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
            images.append(gf.tensor(data.reshape(image_size, image_size, 1)))
    return gf.tensor(images)


def run_epoch(
    model,
    x,
    y,
    loss_fn,
    batch_size,
    epoch,
    epochs,
    optimizer: Optional[Optimizer] = None,
    train: bool = True,
):
    dataset_str = "Train" if train else "Val"
    correct = 0
    total = 0
    curr_loss = 0

    bar = tqdm(
        range(0, len(x), batch_size), desc="{dataset_str} epoch {epoch}/{epochs}"
    )
    for i in bar:
        with set_grad_enabled(train):
            end = min(i + batch_size, len(x))
            p = model(x[i:end].reshape(end - i, -1) / 255)
            target = gf.zeros_like(p)
            target[np.array(range(end - i)), y[i:end, 0]] = 1
            loss = loss_fn(p, target)

            curr_loss = loss.detach().numpy()[0]
            curr_loss += curr_loss * (end - i)
            total += end - i

            correct += (
                np.array(p.argmax(axis=-1) - np.array(y[i:end]).reshape(-1)) == 0
            ).sum()

            bar.set_description(
                f"{dataset_str} epoch {epoch}/{epochs}: loss {(curr_loss/total):.5f}, acc. {correct/total*100:.2f}%"
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    return curr_loss, correct, total


if __name__ == "__main__":
    BATCH_SIZE = 128
    EPOCHS = 10
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
        train_loss, train_correct, num_train_samples = run_epoch(
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
        val_loss, val_correct, num_val_samples = run_epoch(
            model,
            train_x,
            train_y,
            loss_fn,
            BATCH_SIZE,
            epoch,
            EPOCHS,
            optimizer,
            train=False,
        )

    print(f"Train loss: {train_loss/num_train_samples}")
    print(f"Train accuracy: {train_correct/num_train_samples*100}%")
    print(f"Val loss: {val_loss/num_val_samples}")
    print(f"Val accuracy: {val_correct/num_val_samples*100}%")
