"""

Graphical model class
Authors: kkorovin@cs.cmu.edu

"""

from inference import get_algorithm

dflt_algo = {"marginal": "bp", "map": "bp"}


class GraphicalModel:
    def __init__(self, n_nodes, params=None, default_algo=dflt_algo):
        """Constructor

        Arguments:
            n_nodes {int} - number of vertices in graphical model
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

    def __repr__(self):
        return "GraphicalModel:{} on {} nodes".format(
            self.__class__.__name__, self.n_nodes)


class BinaryMRF(GraphicalModel):
    def __init__(self, W, b, struct=None):
        """Constructor of BinaryMRF class.

        Arguments:
            W {np.array} -- (N, N) matrix of pairwise parameters
            b {np.array} -- (N,) vector of unary parameters
        
        Keyword Arguments:
            struct {string or None} -- description of graph structure
                                       (default: {None})
        """
        self.W = W
        self.b = b
        self.struct = struct
        self.n_nodes = len(W)
        self.default_algo = {"marginal": "bp",
                             "map": "bp"}
        # params = {"W": W, "b": b}
        super(BinaryMRF, self).__init__(
            n_nodes=self.n_nodes,
            default_algo=self.default_algo)


if __name__ == "__main__":
    print(get_algorithm("bp"))
