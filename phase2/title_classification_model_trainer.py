import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from gensim.models.doc2vec import Doc2Vec
from utils import TokenizedFileManager, time_it, get_optimal_workers
from constants import BOARDS
from logger import stream_log
from sklearn.model_selection import train_test_split

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)


class TitleClassificationDataset(Dataset):
    def __init__(
        self,
        titles: list[list[str]],
        boards: list[str],
        embedding_model: Doc2Vec,
    ):
        self.titles = titles
        self.boards = boards
        self.embedding_model = embedding_model
        self.board_label_map = {board: idx for idx, board in enumerate(BOARDS)}
        self.num_classes = len(BOARDS)

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx: int):
        title = self.titles[idx]
        board = self.boards[idx]
        title_vector = self.embedding_model.infer_vector(title)
        board_label = self.board_label_map[board]
        # Create one-hot vector
        one_hot = torch.zeros(self.num_classes)
        one_hot[board_label] = 1
        return torch.FloatTensor(title_vector), one_hot


class TitleClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        hidden_size_2: int,
        num_classes: int,
    ):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, num_classes),
        )

    def forward(self, x):
        return self.stack(x)


def parse_tokenized_file(
    tokenized_file_manager: TokenizedFileManager,
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

    return titles, boards


def train_model(
    embedding_model: Doc2Vec,
    train_titles: list[list[str]],
    train_boards: list[str],
    val_titles: list[list[str]],
    val_boards: list[str],
    vector_size: int,
    hidden_size: int,
    hidden_size_2: int,
    batch_size: int = 64,
    epochs: int = 10,
    learning_rate: float = 0.001,
    model_path: str = None,  # Path to existing model
):
    train_dataset = TitleClassificationDataset(
        train_titles, train_boards, embedding_model
    )
    val_dataset = TitleClassificationDataset(val_titles, val_boards, embedding_model)

    worker = get_optimal_workers()
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=worker
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=worker)

    # Initialize model
    classifier = TitleClassifier(
        input_size=vector_size,
        hidden_size=hidden_size,
        hidden_size_2=hidden_size_2,
        num_classes=len(BOARDS),
    )

    # Load existing model if provided
    if model_path:
        checkpoint = torch.load(model_path)
        classifier.load_state_dict(checkpoint["model_state_dict"])
        board_label_map = checkpoint["board_label_map"]
        stream_log.info(f"Loaded model from {model_path}")
    else:
        board_label_map = train_dataset.board_label_map

    classifier = classifier.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        train(classifier, train_loader, optimizer, loss_fn, epoch, epochs)
        validate(classifier, val_loader)

    return classifier, board_label_map


@time_it
def train(
    classifier: TitleClassifier,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.CrossEntropyLoss,
    epoch: int,
    epochs: int,
):
    classifier.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = classifier(inputs)
        # Convert one-hot labels to class indices for CrossEntropyLoss
        _, target_indices = torch.max(labels, 1)
        loss = loss_fn(outputs, target_indices)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == target_indices).sum().item()

    train_accuracy = 100 * correct / total
    stream_log.info(
        f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}, "
        f"Train Accuracy: {train_accuracy:.2f}%"
    )


def validate(classifier: TitleClassifier, val_loader: DataLoader):
    classifier.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = classifier(inputs)
            _, target_indices = torch.max(labels, 1)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == target_indices).sum().item()

    val_accuracy = 100 * val_correct / val_total
    stream_log.info(f"Validation Accuracy: {val_accuracy:.2f}%")


if __name__ == "__main__":
    # Load your trained Doc2Vec model
    embedding_model_path = "data/embedding_model_100_2_5_15_1_0_0.1.d2v"
    embedding_model = Doc2Vec.load(embedding_model_path)

    # Load and prepare data
    tokenized_file_manager = TokenizedFileManager()
    titles, boards = parse_tokenized_file(tokenized_file_manager)
    stream_log.info(f"Total number of titles: {len(titles)}")

    # Split the data into training and validation sets
    train_titles, val_titles, train_boards, val_boards = train_test_split(
        titles, boards, test_size=0.2, random_state=42, stratify=boards
    )

    stream_log.info(f"Training set size: {len(train_titles)}")
    stream_log.info(f"Validation set size: {len(val_titles)}")

    # Train the classifier
    classifier, board_label_map = train_model(
        embedding_model=embedding_model,
        train_titles=train_titles,
        train_boards=train_boards,
        val_titles=val_titles,
        val_boards=val_boards,
        vector_size=100,  # should match embedding model vector size
        hidden_size=50,
        hidden_size_2=30,
        batch_size=64,
        epochs=5,
        model_path="data/title_classifier_50_30.pth",
    )

    # Save the trained classifier
    torch.save(
        {
            "model_state_dict": classifier.state_dict(),
            "board_label_map": board_label_map,
        },
        "data/title_classifier_50_30.pth",
    )
