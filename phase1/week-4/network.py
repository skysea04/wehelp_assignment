from dataclasses import dataclass, field

import activate_functions as AT


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

    @property
    def index(self):
        return (
            self.from_idx.layer,
            self.from_idx.node,
            self.to_idx.layer,
            self.to_idx.node,
        )


@dataclass
class Network:
    layer_node_cnts: list[int]
    activate_function: AT.OutputLayerActivateFunction
    connections: list[Connection] = field(default_factory=list)
    connection_basic_weight: float = 0.5

    def __post_init__(self):
        self.layers: list[Layer] = [
            Layer(i, node_cnt)
            if i == len(self.layer_node_cnts) - 1
            else LayerWithBias(i, node_cnt)
            for i, node_cnt in enumerate(self.layer_node_cnts)
        ]
        self.input_layer = self.layers[0]
        self.output_layer = self.layers[-1]

        self.connection_dict = {
            (
                prev_node.index.layer,
                prev_node.index.node,
                cur_node.index.layer,
                cur_node.index.node,
            ): self.connection_basic_weight
            for prev_layer, cur_layer in zip(self.layers[:-1], self.layers[1:])
            for prev_node in prev_layer.nodes
            for cur_node in (
                cur_layer.nodes[:-1]
                if cur_layer != self.output_layer
                else cur_layer.nodes
            )
        }

        for connection in self.connections:
            if connection.index not in self.connection_dict:
                continue

            self.connection_dict[connection.index] = connection.weight

    def forward(self, input: list[float]) -> list[float]:
        if len(input) != self.input_layer.node_cnt:
            raise ValueError(
                f"Input length must match the number of nodes in the input layer ({self.input_layer.node_cnt})"
            )

        for i, value in enumerate(input):
            self.input_layer.nodes[i].value = value

        for layer in self.layers[1:]:
            is_output_layer = layer == self.output_layer
            nodes = layer.nodes[:-1] if not is_output_layer else layer.nodes
            for node in nodes:
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

            if is_output_layer:
                if self.activate_function == AT.OutputLayerActivateFunction.LINEAR:
                    for node in layer.nodes:
                        node.value = AT.linear(node.value)
                elif self.activate_function == AT.OutputLayerActivateFunction.SIGMOID:
                    for node in layer.nodes:
                        node.value = AT.sigmoid(node.value)
                elif self.activate_function == AT.OutputLayerActivateFunction.SOFTMAX:
                    output_values = [node.value for node in layer.nodes]
                    output_values = AT.softmax(output_values)
                    for i, node in enumerate(layer.nodes):
                        node.value = output_values[i]
                else:
                    raise ValueError(
                        f"Invalid activate function for the output layer: {self.activate_function}"
                    )
            else:
                for node in layer.nodes[:-1]:
                    node.value = AT.ReLU(node.value)

        output = [node.value for node in self.output_layer.nodes]
        self.reset()

        print("Output", output)
        return output

    def reset(self):
        for layer in self.layers:
            layer.reset()
