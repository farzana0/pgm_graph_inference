"""
Base Inference class:
All inference classes (BP, GNN etc) should subclass this class.
Authors: kkorovin@cs.cmu.edu

"""

class Inference:
    def __init__(self, mode):
        if mode not in ["marginal", "MAP"]:
            raise ValueError("Inference mode {} not supported".format(mode))
        self.mode = mode

    def run(self):
        pass