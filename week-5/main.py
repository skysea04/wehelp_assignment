import loss_functions
from activate_functions import ActivateFunctionEnum as AFEnum
from network import Connection, Index, Network

if __name__ == "__main__":
    print("Task 1")
    # Task 1-1
    inputs = [1.5, 0.5]
    excepts = [0.8, 1]
    loss_fn = loss_functions.MeanSquaredError()
    learning_rate = 0.01
    nn = Network(
        layer_node_cnts=[2, 2, 1, 2],
        activate_functions=[AFEnum.RELU, AFEnum.LINEAR, AFEnum.LINEAR],
        connections=[
            Connection(Index(0, 0), Index(1, 0), 0.5),
            Connection(Index(0, 0), Index(1, 1), 0.6),
            Connection(Index(0, 1), Index(1, 0), 0.2),
            Connection(Index(0, 1), Index(1, 1), -0.6),
            Connection(Index(0, 2), Index(1, 0), 0.3),
            Connection(Index(0, 2), Index(1, 1), 0.25),
            Connection(Index(1, 0), Index(2, 0), 0.8),
            Connection(Index(1, 1), Index(2, 0), -0.5),
            Connection(Index(1, 2), Index(2, 0), 0.6),
            Connection(Index(2, 0), Index(3, 0), 0.6),
            Connection(Index(2, 0), Index(3, 1), -0.3),
            Connection(Index(2, 1), Index(3, 0), 0.4),
            Connection(Index(2, 1), Index(3, 1), 0.75),
        ],
    )
    outputs = nn.forward(inputs)
    loss = loss_fn.get_total_loss(outputs, excepts)
    output_losses = loss_fn.get_output_losses()
    nn.backward(output_losses)
    nn.zero_grad(learning_rate)
    nn.print_weights()

    # Task 1-2
    nn = Network(
        layer_node_cnts=[2, 2, 1, 2],
        activate_functions=[AFEnum.RELU, AFEnum.LINEAR, AFEnum.LINEAR],
        connections=[
            Connection(Index(0, 0), Index(1, 0), 0.5),
            Connection(Index(0, 0), Index(1, 1), 0.6),
            Connection(Index(0, 1), Index(1, 0), 0.2),
            Connection(Index(0, 1), Index(1, 1), -0.6),
            Connection(Index(0, 2), Index(1, 0), 0.3),
            Connection(Index(0, 2), Index(1, 1), 0.25),
            Connection(Index(1, 0), Index(2, 0), 0.8),
            Connection(Index(1, 1), Index(2, 0), -0.5),
            Connection(Index(1, 2), Index(2, 0), 0.6),
            Connection(Index(2, 0), Index(3, 0), 0.6),
            Connection(Index(2, 0), Index(3, 1), -0.3),
            Connection(Index(2, 1), Index(3, 0), 0.4),
            Connection(Index(2, 1), Index(3, 1), 0.75),
        ],
    )
    for i in range(1000):
        outputs = nn.forward(inputs)
        loss = loss_fn.get_total_loss(outputs, excepts)
        output_losses = loss_fn.get_output_losses()
        nn.backward(output_losses)
        nn.zero_grad(learning_rate)

    print("Total Loss", loss)

    print("-" * 50)
    print("Task 2")
    # Task 2-1
    inputs = [0.75, 1.25]
    excepts = [1]
    loss_fn = loss_functions.BinaryCrossEntropy()
    learning_rate = 0.1
    nn = Network(
        layer_node_cnts=[2, 2, 1],
        activate_functions=[AFEnum.RELU, AFEnum.SIGMOID],
        connections=[
            Connection(Index(0, 0), Index(1, 0), 0.5),
            Connection(Index(0, 0), Index(1, 1), 0.6),
            Connection(Index(0, 1), Index(1, 0), 0.2),
            Connection(Index(0, 1), Index(1, 1), -0.6),
            Connection(Index(0, 2), Index(1, 0), 0.3),
            Connection(Index(0, 2), Index(1, 1), 0.25),
            Connection(Index(1, 0), Index(2, 0), 0.8),
            Connection(Index(1, 1), Index(2, 0), 0.4),
            Connection(Index(1, 2), Index(2, 0), -0.5),
        ],
    )
    outputs = nn.forward(inputs)
    loss = loss_fn.get_total_loss(outputs, excepts)
    output_losses = loss_fn.get_output_losses()
    nn.backward(output_losses)
    nn.zero_grad(learning_rate)
    nn.print_weights()

    # Task 2-2
    nn = Network(
        layer_node_cnts=[2, 2, 1],
        activate_functions=[AFEnum.RELU, AFEnum.SIGMOID],
        connections=[
            Connection(Index(0, 0), Index(1, 0), 0.5),
            Connection(Index(0, 0), Index(1, 1), 0.6),
            Connection(Index(0, 1), Index(1, 0), 0.2),
            Connection(Index(0, 1), Index(1, 1), -0.6),
            Connection(Index(0, 2), Index(1, 0), 0.3),
            Connection(Index(0, 2), Index(1, 1), 0.25),
            Connection(Index(1, 0), Index(2, 0), 0.8),
            Connection(Index(1, 1), Index(2, 0), 0.4),
            Connection(Index(1, 2), Index(2, 0), -0.5),
        ],
    )
    for i in range(1000):
        outputs = nn.forward(inputs)
        loss = loss_fn.get_total_loss(outputs, excepts)
        output_losses = loss_fn.get_output_losses()
        nn.backward(output_losses)
        nn.zero_grad(learning_rate)

    print("Total Loss", loss)
