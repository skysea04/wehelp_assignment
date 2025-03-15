import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class GenderHeightWeightDataset(Dataset):
    def __init__(self, csv_file: str):
        self.data = pd.read_csv(csv_file)

        self.data["Gender"] = self.data["Gender"].apply(
            lambda x: 1 if x == "Male" else 0
        )

        height_mean = self.data["Height"].mean()
        height_std = self.data["Height"].std()
        self.data["Height"] = (self.data["Height"] - height_mean) / height_std

        weight_mean = self.data["Weight"].mean()
        weight_std = self.data["Weight"].std()
        self.data["Weight"] = (self.data["Weight"] - weight_mean) / weight_std

        self.features = self.data[["Gender", "Height"]].values
        self.labels = self.data[["Weight"]].values

        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)
        self.label_std = weight_std

    def __getitem__(self, idx: int):
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label

    def __len__(self):
        return len(self.data)


class GenderHeightWeightNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(2, 1),
        )

    def forward(self, x):
        logits = self.stack(x)
        return logits


class TitanicDataset(Dataset):
    def __init__(self, csv_file: str):
        self.data = pd.read_csv(csv_file)

        self.data["Pclass_1"] = self.data["Pclass"].apply(lambda x: 1 if x == 1 else 0)
        self.data["Pclass_2"] = self.data["Pclass"].apply(lambda x: 1 if x == 2 else 0)
        self.data["Pclass_3"] = self.data["Pclass"].apply(lambda x: 1 if x == 3 else 0)
        self.data["Sex"] = self.data["Sex"].apply(lambda x: 1 if x == "male" else 0)
        self.data["Age"] = pd.to_numeric(self.data["Age"], errors="coerce").fillna(35)
        self.data["SibSp"] = self.data["SibSp"].astype(int)
        self.data["Parch"] = self.data["Parch"].astype(int)

        self.features = self.data[
            ["Pclass_1", "Pclass_2", "Pclass_3", "Sex", "Age", "SibSp", "Parch"]
        ].values
        self.labels = self.data[["Survived"]].values

        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def __getitem__(self, idx: int):
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label

    def __len__(self):
        return len(self.data)


class TitanicNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(7, 5),
            nn.ReLU(),
            nn.Linear(5, 3),
            nn.ReLU(),
            nn.Linear(3, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        logits = self.stack(x)
        return logits


def task1():
    print("Task 1:")
    dataset = GenderHeightWeightDataset("gender-height-weight.csv")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    model = GenderHeightWeightNetwork()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Training
    print("Training")
    model.train()
    epochs = 5
    for i in range(epochs):
        loss_pounds = 0
        for feature, label in dataloader:
            pred = model(feature)
            loss = loss_fn(pred, label)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_pounds += loss.item() ** 0.5 * dataset.label_std

        loss_pounds /= len(dataloader)
        print(f"Epoch {i+1}: {loss_pounds:>3f}")

    # Evaluating
    print("Evaluating")
    model.eval()
    loss_pounds = 0
    with torch.no_grad():
        for feature, label in dataloader:
            pred = model(feature)
            loss_pounds += loss_fn(pred, label).item() ** 0.5 * dataset.label_std

    loss_pounds /= len(dataloader)
    print(f"Loss Pounds: {loss_pounds:>3f}")


def task2():
    print("Task 2:")
    dataset = TitanicDataset("titanic.csv")
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    model = TitanicNetwork()
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=3e-3)

    # Training
    print("Training")
    model.train()
    epochs = 300
    for i in range(1, epochs + 1):
        for X, y in dataloader:
            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if i % 50 == 0:
            print(f"Epoch {i} last batch loss: {loss:>5f}")

    # Evaluating
    print("\nEvaluating")
    model.eval()
    correct_cnt = 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            for i in range(len(pred)):
                survived = 1 if pred[i] > 0.5 else 0
                correct_cnt += 1 if survived == y[i] else 0

    correct_rate = round(
        correct_cnt / (len(dataloader) * dataloader.batch_size) * 100, 2
    )
    print(f"Correct Rate: {correct_rate}%")


if __name__ == "__main__":
    task1()
    print("-" * 50)
    task2()
