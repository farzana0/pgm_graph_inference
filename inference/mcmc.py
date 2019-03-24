"""

Implements MCMC inference procedures.
Authors: kkorovin@cs.cmu.edu

TODO:
* implement Gibbs sampling
* (or MH in the case Gibbs is not possible)
"""

from inference.core import Inference
import numpy as np

class GibbsSampling(Inference):
	"""Gibbs sampling for binaryMRF.
	"""
	def conditonal(self, i, X):
		'''
			return P(x_i=1|other)
		'''
		def sigmoid(x):
			return 1. / (1 + np.exp(-x))
		tmp = self.W[i, :].dot(X)
		return sigmoid(2 * (tmp + self.u[i]))

	def gibbs_sampling(self, n, burn_in=100, stride=2):
		X = np.array([1 if np.random.rand() < .5 else -1 for i in range(self.d)])
		samples = [np.copy(X)]
		for i in range(stride*n + burn_in-1):
			for j in range(self.d):
				p = self.conditonal(j, X)
				X[j] = +1 if np.random.rand() < p else -1
			samples.append(np.copy(X))
		return np.array(samples[burn_in::stride])

	def function():
		pass

	def run(self, graphs, n):
		samples = []
		for graph in graphs:
			self.W = graph.W
			self.u = graph.b
			self.d = graph.n_nodes

			sample = self.gibbs_sampling(n)
			samples.append(sample)

		return samples

if __name__ == '__main__':
	gibs = GibbsSampling("map")
	W = np.array([[0, -1, 0, 0, 0, 0, 0],
			  [-1, 0, 1.5, 1, 0, 0, 0],
			  [0, 1.5, 0, 0, 1.5, 2, -2],
			  [0, 1, 0, 0, 0, 0, 0],
			  [0, 0, 1.5, 0, 0, 0, 0],
			  [0, 0, 2, 0, 0, 0, 0],
			  [0, 0, -2, 0, 0, 0, 0]])
	u = np.zeros(7)
	from graphical_models.data_structs import BinaryMRF
	graphs = [BinaryMRF(W, u)]
	samples = gibs.run(graphs, 100)
	print(samples[0])