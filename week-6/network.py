import random
from dataclasses import dataclass, field

from activate_functions import ActivateFunctionEnum as AFEnum
from activate_functions import linear, relu, sigmoid, softmax


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

    def iter_nodes(self):
        for node in self.nodes:
            yield node


@dataclass
class LayerWithBias(Layer):
    def __post_init__(self):
        super().__post_init__()
        self.nodes.append(BiasNode(Index(self.layer_idx, self.node_cnt)))

    def reset(self):
        for node in self.nodes[:-1]:
            node.value = 0

    def iter_nodes(self):
        for node in self.nodes[:-1]:
            yield node


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
    activate_functions: list[AFEnum] = field(default_factory=list)
    connections: list[Connection] = field(default_factory=list)
    connection_basic_weight: float = 0.5
    random_weights: bool = False

    def __post_init__(self):
        # create layers
        self.layers: list[Layer] = [
            Layer(i, node_cnt)
            if i == len(self.layer_node_cnts) - 1
            else LayerWithBias(i, node_cnt)
            for i, node_cnt in enumerate(self.layer_node_cnts)
        ]
        self.input_layer = self.layers[0]
        self.output_layer = self.layers[-1]

        # adjust activate functions
        if len(self.activate_functions) < len(self.layers) - 1:
            self.activate_functions += [AFEnum.LINEAR] * (
                len(self.layers) - 1 - len(self.activate_functions)
            )
        elif len(self.activate_functions) > len(self.layers) - 1:
            self.activate_functions = self.activate_functions[: len(self.layers) - 1]

        # adjust connections
        self.connection_dict = {
            (
                prev_node.index.layer,
                prev_node.index.node,
                cur_node.index.layer,
                cur_node.index.node,
            ): random.random() if self.random_weights else self.connection_basic_weight
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

        # create gradients
        self.gradients = {
            connection_index: 0 for connection_index in self.connection_dict
        }

    def print_weights(self) -> list[list[float]]:
        for i, layer in enumerate(self.layers, 1):
            if i == len(self.layers):
                continue

            next_layer = self.layers[i]
            print("Layer", i)
            weights = []
            bias_weights = []
            for next_layer_node in next_layer.iter_nodes():
                node_weights = []
                for node in layer.iter_nodes():
                    node_weights.append(
                        self.connection_dict[
                            (
                                node.index.layer,
                                node.index.node,
                                next_layer_node.index.layer,
                                next_layer_node.index.node,
                            )
                        ]
                    )
                weights.append(node_weights)
                bias_weights.append(
                    self.connection_dict[
                        (
                            layer.nodes[-1].index.layer,
                            layer.nodes[-1].index.node,
                            next_layer_node.index.layer,
                            next_layer_node.index.node,
                        )
                    ]
                )
            print(weights)
            print(bias_weights)

    def reset(self):
        for layer in self.layers:
            layer.reset()

    def forward(self, input: list[float]) -> list[float]:
        self.reset()

        if len(input) != self.input_layer.node_cnt:
            raise ValueError(
                f"Input length must match the number of nodes in the input layer ({self.input_layer.node_cnt})"
            )

        for i, value in enumerate(input):
            self.input_layer.nodes[i].value = value

        for i, layer in enumerate(self.layers[1:]):
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

            if self.activate_functions[i] == AFEnum.LINEAR:
                for node in layer.nodes:
                    node.value = linear.activate(node.value)

            elif self.activate_functions[i] == AFEnum.RELU:
                for node in layer.nodes:
                    node.value = relu.activate(node.value)

            elif self.activate_functions[i] == AFEnum.SIGMOID:
                for node in layer.nodes:
                    node.value = sigmoid.activate(node.value)

            elif self.activate_functions[i] == AFEnum.SOFTMAX:
                output_values = [node.value for node in layer.nodes]
                output_values = softmax.activate(output_values)
                for i, node in enumerate(layer.nodes):
                    node.value = output_values[i]

            else:
                raise ValueError(
                    f"Invalid activate function for the output layer: {self.activate_functions[i]}"
                )

        output = [node.value for node in self.output_layer.nodes]

        return output

    def backward(self, output_losses: list[float]):
        node_gradients = {
            (node.index.layer, node.index.node): 0
            for layer in self.layers
            for node in layer.nodes
        }

        # Set output layer gradients
        for i, node in enumerate(self.output_layer.nodes):
            node_gradients[(node.index.layer, node.index.node)] = output_losses[i]

        for layer_idx in range(len(self.layers) - 1, 0, -1):
            layer = self.layers[layer_idx]
            prev_layer = self.layers[layer_idx - 1]
            activate_function = self.activate_functions[layer_idx - 1]

            # Process each node in current layer
            for current_node in layer.iter_nodes():
                current_grad = node_gradients[
                    (current_node.index.layer, current_node.index.node)
                ]

                # Apply activation function derivative
                if activate_function == AFEnum.LINEAR:
                    activation_grad = linear.derivative(current_node.value)
                elif activate_function == AFEnum.RELU:
                    activation_grad = relu.derivative(current_node.value)
                elif activate_function == AFEnum.SIGMOID:
                    activation_grad = sigmoid.derivative(current_node.value)
                elif activate_function == AFEnum.SOFTMAX:
                    activation_grad = softmax.derivative(current_node.value)
                else:
                    raise ValueError(
                        f"Invalid activation function: {activate_function}"
                    )

                current_grad *= activation_grad

                # Update gradients for incoming connections and nodes
                for prev_node in prev_layer.nodes:
                    connection_index = (
                        prev_node.index.layer,
                        prev_node.index.node,
                        current_node.index.layer,
                        current_node.index.node,
                    )

                    # Update connection gradient
                    self.gradients[connection_index] += current_grad * prev_node.value

                    # Propagate gradient to previous node
                    prev_node_grad = (
                        current_grad * self.connection_dict[connection_index]
                    )
                    node_gradients[(prev_node.index.layer, prev_node.index.node)] += (
                        prev_node_grad
                    )

    def zero_grad(self, learning_rate: float):
        import pprint

        # print("gradients")
        # pprint.pprint(self.gradients)

        for connection_index, weight in self.connection_dict.items():
            self.connection_dict[connection_index] = (
                weight - learning_rate * self.gradients[connection_index]
            )
        # print("connection_dict")
        # pprint.pprint(self.connection_dict)
        self.gradients = {
            connection_index: 0 for connection_index in self.connection_dict
        }
        # print("gradients after zero_grad")
        # pprint.pprint(self.gradients)
