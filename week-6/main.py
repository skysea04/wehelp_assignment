import csv
import random
import statistics

import loss_functions
import torch
from activate_functions import ActivateFunctionEnum as AFEnum
from network import Network

if __name__ == "__main__":
    # Task 1
    # Process data
    data_lst = []
    heights = []
    weights = []
    with open("gender-height-weight.csv", "r") as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            gender = 1 if row["Gender"] == "Male" else 0
            height = float(row["Height"])
            weight = float(row["Weight"])
            heights.append(height)
            weights.append(weight)

            data_lst.append((gender, height, weight))

    height_pstd_dev = statistics.pstdev(heights)
    height_mean = statistics.mean(heights)
    weight_pstd_dev = statistics.pstdev(weights)
    weight_mean = statistics.mean(weights)

    random.shuffle(data_lst)

    inputs = []
    expects = []
    for data in data_lst:
        inputs.append([data[0], (data[1] - height_mean) / height_pstd_dev])
        expects.append([(data[2] - weight_mean) / weight_pstd_dev])

    # Build network
    loss_fn = loss_functions.MeanSquaredError()
    learning_rate = 0.001
    nn = Network(
        layer_node_cnts=[2, 3, 1],
        activate_functions=[AFEnum.RELU, AFEnum.LINEAR],
        random_weights=True,
    )

    # Training
    for i in range(10):
        loss_sum = 0
        for input, expect in zip(inputs, expects):
            outputs = nn.forward(input)
            loss = loss_fn.get_total_loss(outputs, expect)
            output_losses = loss_fn.get_output_losses()
            nn.backward(output_losses)
            nn.zero_grad(learning_rate)

    # Get result
    loss_sum = 0
    for input, expect in zip(inputs, expects):
        outputs = nn.forward(input)
        loss = loss_fn.get_total_loss(outputs, expect)
        loss_sum += loss**0.5 * weight_pstd_dev
    avg_loss = loss_sum / len(inputs)
    print(f"Task 1: {avg_loss} pounds\n")

    # Task 2
    # Process data
    passenger_lst = []
    survived_lst = []
    with open("titanic.csv", "r") as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            if row["Pclass"] == "1":
                p_class = [1, 0, 0]
            elif row["Pclass"] == "2":
                p_class = [0, 1, 0]
            else:
                p_class = [0, 0, 1]
            sex = 1 if row["Sex"] == "male" else 0
            age = float(row["Age"]) if row["Age"] != "" else 35
            sib_sp = int(row["SibSp"])
            parch = int(row["Parch"])

            passenger_lst.append((*p_class, sex, age, sib_sp, parch))

            survived = int(row["Survived"])
            survived_lst.append([survived])

    # Build network
    loss_fn = loss_functions.BinaryCrossEntropy()
    learning_rate = 0.005
    nn = Network(
        layer_node_cnts=[7, 5, 3, 1],
        activate_functions=[AFEnum.RELU, AFEnum.RELU, AFEnum.SIGMOID],
        random_weights=True,
    )
    threshold = 0.5

    # Training
    for i in range(40):
        correct_cnt = 0
        cnter = 0
        for input, expect in zip(passenger_lst, survived_lst):
            output = nn.forward(input)
            loss = loss_fn.get_total_loss(output, expect)
            output_losses = loss_fn.get_output_losses()
            nn.backward(output_losses)
            nn.zero_grad(learning_rate)
            survived = 1 if output[0] > threshold else 0
            if survived == expect[0]:
                correct_cnt += 1

        if i % 10 == 0:
            print(f"epoch {i}: {correct_cnt}")

    # Get result
    correct_cnt = 0
    for input, expect in zip(passenger_lst, survived_lst):
        output = nn.forward(input)
        survived = 1 if output[0] > threshold else 0
        if survived == expect[0]:
            correct_cnt += 1

    correct_rate = correct_cnt / len(passenger_lst)
    print(f"Task 2: {correct_rate}\n")

    # Task 3
    print("Task 3-1:")
    tensor = torch.tensor([[2, 3, 1], [5, -2, 1]])
    print(tensor.shape, tensor.dtype)

    print("Task 3-2:")
    tensor = torch.rand((3, 4, 2))
    print(tensor.shape)
    print(tensor)

    print("Task 3-3:")
    tensor = torch.ones((2, 1, 5))
    print(tensor.shape)
    print(tensor)

    print("Task 3-4:")
    tensor1 = torch.tensor([[1, 2, 4], [2, 1, 3]])
    tensor2 = torch.tensor([[5], [2], [1]])
    tensor = tensor1 @ tensor2
    print(tensor.shape)
    print(tensor)

    print("Task 3-5:")
    tensor1 = torch.tensor([[1, 2], [2, 3], [-1, 3]])
    tensor2 = torch.tensor([[5, 4], [2, 1], [1, -5]])
    tensor = tensor1 * tensor2
    print(tensor.shape)
    print(tensor)
