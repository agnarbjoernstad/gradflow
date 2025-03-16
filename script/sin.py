from gradflow import tensor
from gradflow.nn.modules import linear, container, activation
from gradflow.nn.modules.loss import MSELoss
from gradflow.optim.adam import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os


if __name__ == "__main__":
    np.random.seed(42)

    BATCH_SIZE = 128
    EPOCHS = 200
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

    os.makedirs("figure", exist_ok=True)

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

        scaling_factor = 1.2
        plt.xlim(
            min(train_x_min, val_x_min) * scaling_factor,
            min(train_x_max, val_x_max) * scaling_factor,
        )
        plt.grid()
        plt.ylim(-scaling_factor, scaling_factor)
        plt.scatter(val_x, val_y, label="Val ground truth", s=1)
        plt.scatter(
            val_x,
            model(tensor(val_x).unsqueeze(-1)).detach().numpy(),
            label="Val predicted",
            s=1,
        )
        plt.title(
            f"sin(x): epoch {epoch+1}, train MSE loss: {train_loss:.5f}, val MSE loss: {val_loss:.5f}"
        )
        plt.legend()
        plt.savefig("figure/sin.png")
        plt.clf()
    print(f"Train MSE loss: {train_loss:.5f}, Val MSE loss: {val_loss:.5f}")
