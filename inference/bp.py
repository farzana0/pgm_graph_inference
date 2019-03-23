"""

Approximate inference using Belief Propagation
Here we can rely on some existing library,
for example https://github.com/mbforbes/py-factorgraph
Authors: kkorovin@cs.cmu.edu
         lingxiao@cmu.edu
"""

from core import Inference
import numpy as np 
from scipy.misc import logsumexp


class BeliefPropagation(Inference):
    """
    A special case implementation of BP
    for binary MRFs.
    Exact BP in tree structure only need two passes,
    LBP need multiple passes until convergene. 
    """

    def run_one(self, graph, smooth=0):
        # Asynchronous BP  
        # Sketch of algorithm:
        # -------------------
        # preprocessing:
        # - sort nodes by number of edges
        # Algo:
        # - initialize messages to 1
        # - until convergence or max iters reached:
        #     - for each node in sorted list (fewest edges to most):
        #         - compute outgoing messages to neighbors
        #         - check convergence of messages

        if self.mode == "marginal": # not using log
            sumOp = logsumexp
        else:
            sumOp = np.max
        # storage, W should be symmetric 
        max_iters = 100
        epsilon = 1e-10 # determines when to stop

        row, col = np.where(graph.W)
        n_V, n_E = len(graph.b), len(row)
        # create index dict
        degrees = np.sum(graph.W != 0, axis=0)
        index_bases = np.zeros(n_V, dtype=np.int64)
        for i in range(1, n_V): 
            index_bases[i] = index_bases[i-1] + degrees[i-1]

        neighbors = {i:[] for i in range(n_V)}
        for i,j in zip(row,col): neighbors[i].append(j)
        neighbors = {k: sorted(v) for k, v in neighbors.items()}
        # sort nodes by neighbor size 
        ordered_nodes = np.argsort(degrees)
        # init messages based on graph structure (E, 2)
        # messages are ordered (out messages)
        messages = np.zeros([n_E, 2])  # log
        xij = np.array([[1,-1],[-1,1]])
        xi = np.array([-1, 1])
        for _ in range(max_iters):
            converged = True
            # save old message for checking convergence
            old_messages = messages[:]
            # update messages 
            for i in ordered_nodes:
                neighbor = neighbors[i]
                Jij = graph.W[i][neighbor] # vector 
                bi = graph.b[i]            # scalar
                local_potential = Jij.reshape(-1,1,1)*xij + bi*xi.reshape(-1,1,1) 
                # get in messages product (log)
                in_message_prod = 0
                for j in neighbor:
                    in_message_prod += messages[index_bases[j]+neighbors[j].index(i)]
                for k in range(degrees[i]):
                    j = neighbor[k]
                    messages[index_bases[i]+k] = in_message_prod - (messages[index_bases[j]+neighbors[j].index(i)])
                # update
                messages[index_bases[i]:index_vbases[i]+degrees[i]] = sumOp(messages[index_bases[i]:index_bases[i]+degrees[i]].reshape(degrees[i],2,1) + local_potential, axis=1)

            # check convergence 
            error = (messages - old_messages).mean()
            if error < epsilon: break

        # calculate marginal or map
        probs = np.zeros([n_V, 2])
        for i in range(n_V):
            probs[i] = graph.b[i]*xi
            for j in neighbors[i]:
                probs[i] += messages[index_bases[j]+neighbors[j].index(i)] 
        probs = np.exp(probs)
        # normalize
        if self.mode == 'marginal':
            results = probs / probs.sum(axis=1, keepdims=True)

        if self.mode == 'map':
            results = np.argmax(probs, axis=1)
            results[results==0] = -1

        return results


    def run(self, graphs):
        res = []
        for graph in graphs:
            res.append(self.run_one(graph))
        return res

if __name__ == "__main__":
    bp = BeliefPropagation("marginal")
    # test
