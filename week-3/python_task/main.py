from dataclasses import dataclass, field


@dataclass
class Index:
    layer: int
    node: int


@dataclass
class Node:
    index: Index
    value: float = 0


@dataclass
class BiasNode(Node):
    value: float = 1


@dataclass
class Layer:
    layer_idx: int
    node_cnt: int
    nodes: list[Node] = field(default_factory=list)

    def __post_init__(self):
        for i in range(self.node_cnt):
            self.nodes.append(Node(Index(self.layer_idx, i)))

    def reset(self):
        for node in self.nodes:
            node.value = 0


@dataclass
class LayerWithBias(Layer):
    def __post_init__(self):
        super().__post_init__()
        self.nodes.append(BiasNode(Index(self.layer_idx, self.node_cnt)))

    def reset(self):
        for node in self.nodes[:-1]:
            node.value = 0


@dataclass
class Connection:
    from_idx: Index
    to_idx: Index
    weight: float


@dataclass
class Network:
    layer_node_cnts: list[int]
    connections: list[Connection] = field(default_factory=list)

    def __post_init__(self):
        self.connection_dict = {
            (
                connection.from_idx.layer,
                connection.from_idx.node,
                connection.to_idx.layer,
                connection.to_idx.node,
            ): connection.weight
            for connection in self.connections
        }

        self.layers: list[Layer] = [
            Layer(i, node_cnt)
            if i == len(self.layer_node_cnts) - 1
            else LayerWithBias(i, node_cnt)
            for i, node_cnt in enumerate(self.layer_node_cnts)
        ]

        self.input_layer = self.layers[0]
        self.output_layer = self.layers[-1]

    def forward(self, input: list[float]) -> list[float]:
        if len(input) != self.input_layer.node_cnt:
            raise ValueError(
                f"Input length must match the number of nodes in the input layer ({self.input_layer.node_cnt})"
            )

        for i, value in enumerate(input):
            self.input_layer.nodes[i].value = value

        for layer in self.layers[1:]:
            for node in layer.nodes:
                for prev_node in self.layers[layer.layer_idx - 1].nodes:
                    node.value += prev_node.value * self.connection_dict.get(
                        (
                            prev_node.index.layer,
                            prev_node.index.node,
                            node.index.layer,
                            node.index.node,
                        ),
                        0,
                    )

        output = [node.value for node in self.output_layer.nodes]
        self.reset()

        return output

    def reset(self):
        for layer in self.layers:
            layer.reset()


if __name__ == "__main__":
    print("Model 1")
    nn = Network(
        layer_node_cnts=[2, 2, 1],
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

    outputs = nn.forward([1.5, 0.5])
    print(outputs)

    outputs = nn.forward([0, 1])
    print(outputs)

    print("\nModel 2")
    nn = Network(
        layer_node_cnts=[2, 2, 1, 2],
        connections=[
            Connection(Index(0, 0), Index(1, 0), 0.5),
            Connection(Index(0, 0), Index(1, 1), 0.6),
            Connection(Index(0, 1), Index(1, 0), 1.5),
            Connection(Index(0, 1), Index(1, 1), -0.8),
            Connection(Index(0, 2), Index(1, 0), 0.3),
            Connection(Index(0, 2), Index(1, 1), 1.25),
            Connection(Index(1, 0), Index(2, 0), 0.6),
            Connection(Index(1, 1), Index(2, 0), -0.8),
            Connection(Index(1, 2), Index(2, 0), 0.3),
            Connection(Index(2, 0), Index(3, 0), 0.5),
            Connection(Index(2, 0), Index(3, 1), -0.4),
            Connection(Index(2, 1), Index(3, 0), 0.2),
            Connection(Index(2, 1), Index(3, 1), 0.5),
        ],
    )

    outputs = nn.forward([0.75, 1.25])
    print(outputs)

    outputs = nn.forward([-1, 0.5])
    print(outputs)
