"""

Graphical model class
Authors: kkorovin@cs.cmu.edu

"""

from inference import get_algorithm

dflt_algo = {"marginal": "bp", "map": "bp"}

class GraphicalModel:
    def __init__(self, params, default_algo=dflt_algo):
        """Constructor

        Arguments:
            params {dictionary<str,np.array> or None} -- parameters of the model

        Keyword Arguments:
            default_algo {dict} -- default inference methods to use,
            unless they are overriden in corresponding methods
            (default: {dflt_algo})
        """
        self.algo_marginal = default_algo["marginal"]
        self.algo_map = default_algo["map"]

    def set_ground_truth(self):
        pass

    def get_marginal(self, i, algo=None):
        if algo is None:
            algo = self.algo_marginal
        algo_obj = get_algorithm(algo)
        # algo_obj(get_map)

    def get_map(self, algo=None):
        if algo is None:
            algo = self.algo_map
        algo_obj = get_algorithm(algo)
        # algo_obj(get_map)


class BinaryMRF(GraphicalModel):
    pass


if __name__ == "__main__":
    print(get_algorithm("bp"))