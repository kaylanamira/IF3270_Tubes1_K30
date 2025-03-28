import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self, model):
        self.model = model
        self.layers = model.layers
        self.graph = nx.DiGraph()
        self.pos = {}
        self.node_colors = {}

    def get_network_graph(self):
        n_input = self.layers[0].W.shape[1]
        for i in range(n_input):
            input_id = f"I{i+1}"
            self.graph.add_node(input_id)
            self.node_colors[input_id] = "orange"
            self.pos[input_id] = (-2, -i * 2)

        for layer_idx, layer in enumerate(self.layers):
            n_neuron = layer.W.shape[0]
            n_inputs = layer.W.shape[1]

            bias_id = f"B{layer_idx+1}"
            self.graph.add_node(bias_id)
            self.node_colors[bias_id] = "green"
            lowest_y = -n_neuron * 2 
            self.pos[bias_id] = (layer_idx * 2 - 1, lowest_y - 2)

            for neuron_idx in range(n_neuron):
                # Neuron ID
                if layer_idx == len(self.layers) - 1:
                    neuron_id = f"O{neuron_idx+1}"
                else:
                    neuron_id = f"L{layer_idx+1}_N{neuron_idx+1}"

                self.graph.add_node(neuron_id)
                self.pos[neuron_id] = (layer_idx * 2, -neuron_idx * 2)
                self.node_colors[neuron_id] = "skyblue"

                # Edges from previous layer or input
                for prev_neuron_idx in range(n_inputs):
                    prev_neuron_id = (
                        f"L{layer_idx}_N{prev_neuron_idx+1}" if layer_idx > 0 else f"I{prev_neuron_idx+1}"
                    )
                    weight = round(layer.W[neuron_idx, prev_neuron_idx], 2)
                    grad = round(layer.grad_W[neuron_idx, prev_neuron_idx], 2) if hasattr(layer, "grad_W") else 0.0
                    self.graph.add_edge(prev_neuron_id, neuron_id, weight=weight, grad=grad)

                # Bias edge
                bias_weight = round(layer.b[neuron_idx], 2)
                bias_grad = round(layer.grad_b[neuron_idx], 2) if hasattr(layer, "grad_b") else 0.0
                self.graph.add_edge(bias_id, neuron_id, weight=bias_weight, grad=bias_grad)

    def plot_neural_network(self):
        self.get_network_graph()
        node_color_list = [self.node_colors[n] for n in self.graph.nodes]

        plt.figure(figsize=(12, 8))
        nx.draw(
            self.graph,
            self.pos,
            with_labels=True,
            node_color=node_color_list,
            node_size=1000,
            edge_color="gray"
        )

        edge_labels = {
            (src, dest): f"w={data['weight']}, g={data['grad']}"
            for src, dest, data in self.graph.edges(data=True)
        }
        nx.draw_networkx_edge_labels(
            self.graph, self.pos, edge_labels=edge_labels,
            font_color="red", font_size=10, label_pos=0.2
        )
        plt.show()
