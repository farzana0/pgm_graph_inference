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

    def set_ground_truth(self, marginal_est=None, map_est=None):
        """ Setting labels:
        To be used when instantiating
        a model from saved parameters
        """
        self.marginal = marginal_est
        self.map = map_est

    # Running inference with/without Inference object
    def get_marginals(self, algo_obj=None, algo=None):
        if algo_obj is None:
            if algo is None:
                algo = self.algo_marginal
            algo_obj = get_algorithm(algo)
        inf_res = algo_obj.run(self, mode="marginal")
        return inf_res

    def get_map(self, algo_obj=None, algo=None):
        if algo_obj is None:
            if algo is None:
                algo = self.algo_map
            algo_obj = get_algorithm(algo)
        inf_res = algo_obj.run(self, mode="map")
        return inf_res


class BinaryMRF(GraphicalModel):
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.default_algo = {"marginal": "bp",
                             "map": "bp"}
        params = {"W": W, "b": b}
        super(BinaryMRF, self).__init__(params,
                                        self.default_algo)

if __name__ == "__main__":
    print(get_algorithm("bp"))
