import algopy
import numpy as np

from rbf_network.functions.gaussian_function import GaussianFunction

graph = algopy.CGraph()

func = GaussianFunction()
func.center = [0, 0]
func.parameters = [1]

x = np.array([1, 1], dtype=float)
fx = algopy.Function(x)
fy = (lambda e: func.value(e))(fx)

graph.trace_off()
graph.independentFunctionList = [fx]
graph.dependentFunctionList = [fy]

print('gradient=', graph.gradient(x))
