import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from utils import (
    TokenizedFileManager,
    time_it,
    get_optimal_workers,
    TitleClassificationResultFileManager,
)
from gensim.models.doc2vec import Doc2Vec
from constants import BOARDS
from logger import stream_log
from sklearn.model_selection import train_test_split
import os
import numpy as np
from tqdm import tqdm


class TitleClassificationDataset(Dataset):
    def __init__(
        self,
        vectors: np.ndarray,
        boards: list[str],
    ):
        self.vectors = torch.FloatTensor(vectors)
        self.boards = boards
        self.board_label_map = {board: idx for idx, board in enumerate(BOARDS)}
        self.num_classes = len(BOARDS)

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, idx: int):
        vector = self.vectors[idx]
        board = self.boards[idx]
        board_label = self.board_label_map[board]
        one_hot = torch.zeros(self.num_classes)
        one_hot[board_label] = 1
        return vector, one_hot


class TitleClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        num_classes: int,
    ):
        super().__init__()
        self.stack = nn.Sequential(
            *[
                layer
                for i in range(len(hidden_sizes))
                for layer in [
                    nn.Linear(
                        input_size if i == 0 else hidden_sizes[i - 1], hidden_sizes[i]
                    ),
                    nn.ReLU(),
                ]
            ]
            + [nn.Linear(hidden_sizes[-1], num_classes)],
        )

    def forward(self, x):
        return self.stack(x)


def parse_tokenized_file(
    tokenized_file_manager: TokenizedFileManager, data_size: int
) -> tuple[list[list[str]], list[str]]:
    titles = []
    boards = []

    for line in tokenized_file_manager.read_titles():
        words = line.split(",")
        if len(words) < 2:
            continue
        board = words[0]
        if board in BOARDS:
            titles.append(words[1:])
            boards.append(board)

        if len(titles) >= data_size:
            break

    return titles, boards


def train_model(
    train_vectors: np.ndarray,
    train_boards: list[str],
    val_vectors: np.ndarray,
    val_boards: list[str],
    vector_size: int,
    hidden_sizes: list[int],
    batch_size: int = 64,
    epochs: int = 10,
    learning_rate: float = 0.001,
):
    train_dataset = TitleClassificationDataset(train_vectors, train_boards)
    val_dataset = TitleClassificationDataset(val_vectors, val_boards)

    worker = get_optimal_workers()
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=worker
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=worker)

    classifier = TitleClassifier(
        input_size=vector_size,
        hidden_sizes=hidden_sizes,
        num_classes=len(BOARDS),
    )

    model_path = f"data/title_classifier_{vector_size}_{'_'.join(map(str, hidden_sizes))}_{len(BOARDS)}_{data_size}.pth"

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        classifier.load_state_dict(checkpoint["model_state_dict"])
        board_label_map = checkpoint["board_label_map"]
        stream_log.info(f"Loaded model from {model_path}")
    else:
        board_label_map = train_dataset.board_label_map

    # classifier = classifier.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        train_accuracy = train(
            classifier, train_loader, optimizer, loss_fn, epoch, epochs
        )
        validation_accuracy = validate(classifier, val_loader)

        if epoch % 5 == 0:
            torch.save(
                {
                    "model_state_dict": classifier.state_dict(),
                    "board_label_map": board_label_map,
                },
                model_path,
            )

    return train_accuracy, validation_accuracy


@time_it
def train(
    classifier: TitleClassifier,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.CrossEntropyLoss,
    epoch: int,
    epochs: int,
) -> float:
    classifier.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

    for inputs, labels in pbar:
        # inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = classifier(inputs)
        _, target_indices = torch.max(labels, 1)
        loss = loss_fn(outputs, target_indices)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == target_indices).sum().item()

    train_accuracy = round(correct / total, 3)
    stream_log.info(
        f"Epoch [{epoch}/{epochs}], Loss: {total_loss / len(train_loader):.4f}, "
        f"Train Accuracy: {train_accuracy}"
    )

    return train_accuracy


def validate(classifier: TitleClassifier, val_loader: DataLoader) -> float:
    classifier.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            # inputs, labels = inputs.to(device), labels.to(device)
            outputs = classifier(inputs)
            _, target_indices = torch.max(labels, 1)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == target_indices).sum().item()

    val_accuracy = round(val_correct / val_total, 3)
    stream_log.info(f"Validation Accuracy: {val_accuracy}")

    return val_accuracy


if __name__ == "__main__":
    embedding_model_path = "data/embedding_model_30_2_5_10_1_0_0.d2v"
    embedding_model = Doc2Vec.load(embedding_model_path)

    # Load and prepare data
    data_size = 1000000
    titles, boards = parse_tokenized_file(TokenizedFileManager(), data_size)
    stream_log.info(f"Total number of titles: {len(titles)}")

    # Split the data into training and validation sets
    train_titles, val_titles, train_boards, val_boards = train_test_split(
        titles,
        boards,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=boards,
    )

    # Precompute embeddings
    stream_log.info("Inferring vectors for training and validation sets...")
    train_vectors = np.array(
        [embedding_model.infer_vector(title) for title in train_titles]
    )
    val_vectors = np.array(
        [embedding_model.infer_vector(title) for title in val_titles]
    )
    stream_log.info("Vector inference complete.")

    stream_log.info(f"Training set size: {len(train_titles)}")
    stream_log.info(f"Validation set size: {len(val_titles)}")
    stream_log.info("-" * 50)

    vector_size = 30
    hidden_sizes = [1200, 1200]
    epochs = 15

    # Train the classifier
    train_accuracy, validation_accuracy = train_model(
        train_vectors=train_vectors,
        train_boards=train_boards,
        val_vectors=val_vectors,
        val_boards=val_boards,
        vector_size=vector_size,
        hidden_sizes=hidden_sizes,
        batch_size=32,
        epochs=epochs,
        learning_rate=0.001,
    )

    # Initialize result file manager with data_size
    title_classification_result_file_manager = TitleClassificationResultFileManager(
        data_size=data_size
    )
    title_classification_result_file_manager.init_file()
    title_classification_result_file_manager.edit_title_classification_result(
        layers=f"{vector_size}_{'_'.join(map(str, hidden_sizes))}_{len(BOARDS)}_{data_size}",
        epochs=epochs,
        train_accuracy=train_accuracy,
        validation_accuracy=validation_accuracy,
    )
