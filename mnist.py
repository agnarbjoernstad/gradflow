from gradflow.nn.modules import linear, module, container, activation
from gradflow.nn.modules.loss import MSELoss, CrossEntropyLoss
from gradflow.autograd.grad_mode import no_grad
from gradflow.nn.functional import relu, softmax
import gzip
import matplotlib.pyplot as plt
import numpy as np
import gradflow as gf
import cv2
import time
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
    lr = 0.01

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
        for i in tqdm(range(0, len(train_x), batch_size)):
            end = min(i + batch_size, len(train_x))
            now = time.time()
            p = model(train_x[i:end].reshape(end - i, -1) / 255)
            target = gf.zeros_like(p)
            target[np.array(range(end - i)), train_y[i:end, 0]] = 1
            loss = loss_fn(p, target)
            now_backward = time.time()
            loss.backward()
            time_backward = time.time() - now_backward

            losses.append(float(loss))
            correct += (
                np.array(p.argmax(axis=-1) - np.array(train_y[i:end]).reshape(-1)) == 0
            ).sum()
            total += end - i
            curr_train_loss += float(loss) * (end - i)

            # print(f"Loss: {loss}")  # , p: {p}, y: {train_y[i]}")
            # print("p", p, "y", train_y[i])
            with no_grad():
                for l in model:
                    if hasattr(l, "weight"):
                        s = l.weight.shape
                        l.weight = l.weight - lr * l.weight.grad
                        l.weight = l.weight.reshape(s)
                        l.weight.grad = None
                    if hasattr(l, "bias"):
                        s = l.bias.shape
                        l.bias = l.bias - lr * l.bias.grad.sum(0)
                        l.bias = l.bias.reshape(s)
                        l.bias.grad = None

            # if i % 500 == 0:
            #    print(f"Epoch: {epoch}, Loss: {loss}, Accuracy: {correct/total}")

            #    # Plot the latest prediction and the probability distribution with opencv.
            #    # image = np.asarray(train_x[i]).squeeze()
            #    #cv2.imwrite("image.png", train_x[i])

        with no_grad():
            for i in tqdm(range(0, len(val_x), batch_size)):
                end = min(i + batch_size, len(val_x))
                now = time.time()
                p = model(val_x[i:end].reshape(end - i, -1) / 255)
                target = gf.zeros_like(p)
                target[np.array(range(end - i)), val_y[i:end, 0]] = 1
                loss = loss_fn(p, target)

                val_losses.append(float(loss) / (end - i))
                val_correct += (
                    np.array(p.argmax(axis=-1) - np.array(val_y[i:end]).reshape(-1))
                    == 0
                ).sum()
                val_total += end - i
                curr_val_loss += float(loss) * (end - i)
        loss = curr_train_loss / total
        val_loss = curr_val_loss / val_total
        print(
            f"Epoch: {epoch}, loss {loss}, val loss: {val_loss}, accuracy {correct/total}, val accuracy: {val_correct/val_total}"
        )
