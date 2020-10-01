import random
import math
import json
import numpy as np
from FileParser import FileParser as fp
from MultilayerPerceptron import MultilayerPerceptron
from Graph import Graph
class ParityResolver:

    with open('settings.json') as config:

        configuration = json.load(config)

        input_nodes_qty = 35
        hidden_nodes_qty = configuration['multilayer_hidden_nodes_parity']
        output_nodes_qty = 1
        lr = configuration['multilayer_lr']
        training_qty = configuration['multilayer_training_qty']
        max_training_epochs = configuration['multilayer_max_training_epochs']

        pb_entries = fp.mlp_entries_parser(7,5)
        pb_targets = [('0'), ('1'), ('0'), ('1'), ('0'), ('1'), ('0'), ('1'), ('0'), ('1')]
        mp = MultilayerPerceptron(pb_entries, pb_targets, input_nodes_qty, hidden_nodes_qty, output_nodes_qty, training_qty, lr,max_training_epochs)

        # for i in range(max_training_epochs):
        #     x = random.randint(0,training_qty)
        #     mp.train(np.matrix(pb_entries[x]).transpose(),np.matrix(pb_targets[x]).transpose())

        mp.train()
        # plot_entries = []
        for pb_e in pb_entries:
            # plot_entries.append(mp.feed_forward(np.matrix(pb_e).transpose()).item(0))
            print(mp.feed_forward(np.matrix(pb_e).transpose()))

        # Graph.graph_multilayer_perceptron(plot_entries)




