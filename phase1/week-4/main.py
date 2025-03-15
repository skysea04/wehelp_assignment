import loss_functions
from activate_functions import OutputLayerActivateFunction
from network import Connection, Index, Network

if __name__ == "__main__":
    print("Regression Tasks")
    nn = Network(
        layer_node_cnts=[2, 2, 2],
        activate_function=OutputLayerActivateFunction.LINEAR,
        connections=[
            Connection(Index(0, 0), Index(1, 0), 0.5),
            Connection(Index(0, 0), Index(1, 1), 0.6),
            Connection(Index(0, 1), Index(1, 0), 0.2),
            Connection(Index(0, 1), Index(1, 1), -0.6),
            Connection(Index(0, 2), Index(1, 0), 0.3),
            Connection(Index(0, 2), Index(1, 1), 0.25),
            Connection(Index(1, 0), Index(2, 0), 0.8),
            Connection(Index(1, 0), Index(2, 1), 0.4),
            Connection(Index(1, 1), Index(2, 0), -0.5),
            Connection(Index(1, 1), Index(2, 1), 0.5),
            Connection(Index(1, 2), Index(2, 0), 0.6),
            Connection(Index(1, 2), Index(2, 1), -0.25),
        ],
    )
    outputs = nn.forward([1.5, 0.5])
    excepts = [0.8, 1]
    print("Total Loss", loss_functions.mean_squared_error(outputs, excepts))

    outputs = nn.forward([0, 1])
    excepts = [0.5, 0.5]
    print("Total Loss", loss_functions.mean_squared_error(outputs, excepts))

    print("-" * 50)
    print("Binary Classification Tasks")
    nn = Network(
        layer_node_cnts=[2, 2, 1],
        activate_function=OutputLayerActivateFunction.SIGMOID,
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
    outputs = nn.forward([0.75, 1.25])
    excepts = [1]
    print("Total Loss", loss_functions.binary_cross_entropy(outputs, excepts))

    outputs = nn.forward([-1, 0.5])
    excepts = [0]
    print("Total Loss", loss_functions.binary_cross_entropy(outputs, excepts))

    print("-" * 50)
    print("Multi-Label Classification Tasks")
    nn = Network(
        layer_node_cnts=[2, 2, 3],
        activate_function=OutputLayerActivateFunction.SIGMOID,
        connections=[
            Connection(Index(0, 0), Index(1, 0), 0.5),
            Connection(Index(0, 0), Index(1, 1), 0.6),
            Connection(Index(0, 1), Index(1, 0), 0.2),
            Connection(Index(0, 1), Index(1, 1), -0.6),
            Connection(Index(0, 2), Index(1, 0), 0.3),
            Connection(Index(0, 2), Index(1, 1), 0.25),
            Connection(Index(1, 0), Index(2, 0), 0.8),
            Connection(Index(1, 0), Index(2, 1), 0.5),
            Connection(Index(1, 0), Index(2, 2), 0.3),
            Connection(Index(1, 1), Index(2, 0), -0.4),
            Connection(Index(1, 1), Index(2, 1), 0.4),
            Connection(Index(1, 1), Index(2, 2), 0.75),
            Connection(Index(1, 2), Index(2, 0), 0.6),
            Connection(Index(1, 2), Index(2, 1), 0.5),
            Connection(Index(1, 2), Index(2, 2), -0.5),
        ],
    )
    outputs = nn.forward([1.5, 0.5])
    excepts = [1, 0, 1]
    print("Total Loss", loss_functions.binary_cross_entropy(outputs, excepts))

    outputs = nn.forward([0, 1])
    excepts = [1, 1, 0]
    print("Total Loss", loss_functions.binary_cross_entropy(outputs, excepts))

    print("-" * 50)
    print("Multi-Class Classification Tasks")
    nn = Network(
        layer_node_cnts=[2, 2, 3],
        activate_function=OutputLayerActivateFunction.SOFTMAX,
        connections=[
            Connection(Index(0, 0), Index(1, 0), 0.5),
            Connection(Index(0, 0), Index(1, 1), 0.6),
            Connection(Index(0, 1), Index(1, 0), 0.2),
            Connection(Index(0, 1), Index(1, 1), -0.6),
            Connection(Index(0, 2), Index(1, 0), 0.3),
            Connection(Index(0, 2), Index(1, 1), 0.25),
            Connection(Index(1, 0), Index(2, 0), 0.8),
            Connection(Index(1, 0), Index(2, 1), 0.5),
            Connection(Index(1, 0), Index(2, 2), 0.3),
            Connection(Index(1, 1), Index(2, 0), -0.4),
            Connection(Index(1, 1), Index(2, 1), 0.4),
            Connection(Index(1, 1), Index(2, 2), 0.75),
            Connection(Index(1, 2), Index(2, 0), 0.6),
            Connection(Index(1, 2), Index(2, 1), 0.5),
            Connection(Index(1, 2), Index(2, 2), -0.5),
        ],
    )
    outputs = nn.forward([1.5, 0.5])
    excepts = [1, 0, 0]
    print("Total Loss", loss_functions.categorical_cross_entropy(outputs, excepts))

    outputs = nn.forward([0, 1])
    excepts = [0, 0, 1]
    print("Total Loss", loss_functions.categorical_cross_entropy(outputs, excepts))
