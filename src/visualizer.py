import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self, model):
        self.layers = model.layers
        self.graph = nx.DiGraph()
        self.pos = {}
        self.node_colors = {}

    def get_network_graph(self):

        for layer_idx, layer in enumerate(self.layers):

            if (layer_idx < layer.n_neuron):
                bias_id = f"B{layer_idx+1}"
                self.graph.add_node(bias_id)
                self.node_colors[bias_id] = "green" 
                self.pos[bias_id] = (layer_idx * 2, -layer.n_neuron * 2)

            for neuron_idx, neuron in enumerate(layer.neurons):
                if (layer_idx == len(self.layers) - 1):
                    neuron_id = f"O{neuron_idx+1}"
                else:
                    neuron_id = f"L{layer_idx+1}_N{neuron_idx+1}"

                self.graph.add_node(neuron_id)
                self.pos[neuron_id] = (layer_idx * 2, -neuron_idx * 2 )
                self.node_colors[neuron_id] = "skyblue" 

                for prev_neuron_idx in range(neuron.n_in):
                    prev_neuron_id = f"L{layer_idx}_N{prev_neuron_idx+1}"
                    self.graph.add_edge(prev_neuron_id, neuron_id, weight=round(neuron.w[prev_neuron_idx].data, 2), grad=round(neuron.w[prev_neuron_idx].grad, 2))

                if (layer_idx>0):
                    neuron_bias_id = f"B{layer_idx}"
                    self.graph.add_edge(neuron_bias_id, neuron_id, weight=round(neuron.b.data, 2), grad=round(neuron.b.grad, 2))

    def plot_neural_network(self):
        self.get_network_graph()
        node_color_list = [self.node_colors[n] for n in self.graph.nodes]

        plt.figure(figsize=(12, 8))
        nx.draw(self.graph, self.pos, with_labels=True, node_color=node_color_list, node_size=1000, edge_color="gray")

        edge_labels = {(src, dest): f"w={data['weight']}, g={data['grad']}" for src, dest, data in self.graph.edges(data=True)}
        nx.draw_networkx_edge_labels(self.graph, self.pos, edge_labels=edge_labels, font_color="red", font_size=10, label_pos=0.2)
        
        plt.show()