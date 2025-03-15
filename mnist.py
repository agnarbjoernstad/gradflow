from gradflow.nn.modules import linear, module, container, activation
from gradflow.nn.modules.loss import MSELoss, CrossEntropyLoss
from gradflow.autograd.grad_mode import no_grad
from gradflow.nn.functional import relu, softmax
from gradflow.optim.adam import Adam
import gzip
import matplotlib.pyplot as plt
import numpy as np
import gradflow as gf
import cv2
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


if __name__ == "__main__":
    mnist_path = "data/mnist"
    train_labels_path = f"{mnist_path}/train-labels-idx1-ubyte.gz"
    train_x_path = f"{mnist_path}/train-images-idx3-ubyte.gz"
    val_labels_path = f"{mnist_path}/t10k-labels-idx1-ubyte.gz"
    val_x_path = f"{mnist_path}/t10k-images-idx3-ubyte.gz"

    train_x = read_images(train_x_path)
    train_y = read_labels(train_labels_path)
    val_x = read_images(val_x_path)
    val_y = read_labels(val_labels_path)

    model = container.Sequential(
        linear.Linear(784, 128),
        activation.ReLU(),
        linear.Linear(128, 128),
        activation.ReLU(),
        linear.Linear(128, 10),
        activation.Softmax(),
    )
    loss_fn = CrossEntropyLoss()
    lr = 0.001
    optimizer = Adam(model.parameters(), lr=lr)

    losses = []
    val_losses = []
    accuracy = []
    val_accuracy = []
    batch_size = 128
    epochs = 1000

    for epoch in range(epochs):
        correct = 0
        total = 0
        val_correct = 0
        val_total = 0
        curr_train_loss = 0
        curr_val_loss = 0
        train_bar = tqdm(
            range(0, len(train_x), batch_size), desc="Train epoch {epoch}/{epochs}"
        )
        for i in train_bar:
            optimizer.zero_grad()
            end = min(i + batch_size, len(train_x))
            p = model(train_x[i:end].reshape(end - i, -1) / 255)
            target = gf.zeros_like(p)
            target[np.array(range(end - i)), train_y[i:end, 0]] = 1
            loss = loss_fn(p, target)
            loss.backward()

            curr_loss = loss.detach().numpy()[0]
            losses.append(curr_loss)
            curr_train_loss += curr_loss * (end - i)
            total += end - i

            correct += (
                np.array(p.argmax(axis=-1) - np.array(train_y[i:end]).reshape(-1)) == 0
            ).sum()
            train_bar.set_description(
                f"Train epoch {epoch}/{epochs}: loss {(curr_train_loss/total):.5f}, acc. {correct/total*100:.2f}%"
            )

            optimizer.step()

        with no_grad():
            val_bar = tqdm(
                range(0, len(val_x), batch_size), desc="Val epoch {epoch}/{epochs}"
            )
            for i in val_bar:
                end = min(i + batch_size, len(val_x))
                p = model(val_x[i:end].reshape(end - i, -1) / 255)
                target = gf.zeros_like(p)
                target[np.array(range(end - i)), val_y[i:end, 0]] = 1
                loss = loss_fn(p, target)

                curr_loss = loss.detach().numpy()[0]
                val_losses.append(curr_loss)
                val_correct += (
                    np.array(p.argmax(axis=-1) - np.array(val_y[i:end]).reshape(-1))
                    == 0
                ).sum()
                val_total += end - i
                curr_val_loss += curr_loss * (end - i)

                val_bar.set_description(
                    f"Val epoch {epoch}/{epochs}: loss {(curr_val_loss/val_total):.5f} acc. {val_correct/val_total*100:.2f}%"
                )
        loss = curr_train_loss / total
        val_loss = curr_val_loss / val_total
