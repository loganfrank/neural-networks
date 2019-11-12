## Basic Python libraries
import os
import sys
from collections import OrderedDict

def convert_data_parallel(network):
    state_dict = OrderedDict()
    for k, v in network.state_dict().items():
        name = k[7:]
        state_dict[name] = v
    return state_dict