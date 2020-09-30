import random
import math
import numpy as np
from MultilayerPerceptron import MultilayerPerceptron
import json




class XorResolver:
    with open('settings.json') as config:

        configuration = json.load(config)

        input_nodes_qty = 2
        hidden_nodes_qty = 2
        output_nodes_qty = 1
        lr = configuration['multilayer_lr']

        pa_entries = [('1 0'),('0 1'), ('0 0'), ('1 1')]
        pa_targets = [('1'), ('1'), ('0'), ('0')]
        mp = MultilayerPerceptron(pa_entries, pa_targets, input_nodes_qty,hidden_nodes_qty,output_nodes_qty,lr)

        # for i in range(10000):
        #     x = random.randint(0,3)
        #     mp.train(np.matrix(pa_entries[x]).transpose(),np.matrix(pa_targets[x]).transpose())

        mp.train()
        print(mp.feed_forward(np.matrix(('1 0')).transpose()))
        print(mp.feed_forward(np.matrix(('0 1')).transpose()))
        print(mp.feed_forward(np.matrix(('0 0')).transpose()))
        print(mp.feed_forward(np.matrix(('1 1')).transpose()))
