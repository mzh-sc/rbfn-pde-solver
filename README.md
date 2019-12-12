# rbfn-pde-solver

rbfn-pde-solver is a software library to solve boundary value and inverse problems using radial basis function networks as an approximation of the solution.

The mathematical background of rbfn-pde-solver is described in [Solving boundary value problems of mathematical physics using radial basis function networks](https://link.springer.com/article/10.1134/S0965542517010079).

Originally, the library was written in C# (see https://github.com/GitMaximZh/rbf-pde-solver-csharp).

### Tech

The library uses 
- [TensorFlow](https://www.tensorflow.org/) for automatic differentiation and solving optimization problems
- [NumPy](www.numpy.org/)
- [Matplotlib](https://matplotlib.org/) 

### Using

Import `solver` package:
```python
import solver as ps
```

Create an approximation model (Radial Basis Function Network):
```python
model = ps.Model()

model.add_rbf(0.2, 'gaussian', [0, 0], [1])
model.add_rbf(1.2, 'gaussian', [0.5, 1], [1.2])
model.add_rbf(-0.5, 'gaussian', [0, 0.7], [0.4])

model.compile()
```

Describe the problem:
```python
problem = ps.Problem()

def equation(y, x):
  h = tf.hessians(y(x), x)[0]
  return h[0][0] + h[1][1]

problem.add_constrain('equation',
                      equation,
                      lambda x: tf.sin(math.pi * x[0]) * tf.sin(math.pi * x[1]))
problem.add_constrain('bc1',
                      lambda y, x: y(x),
                      lambda x: 0)
                      
problem.compile()
```

Configure problem solver:
```python
solver = ps.Solver(problem, model)

solver.set_control_points('equation', 1,
                          ps.uniform_points_2d(0.1, 0.9, 10, 0.1, 0.9, 10))
solver.set_control_points('bc1', 100,
                          ps.uniform_points_2d(0, 1, 10, 0, 0, 1) +
                          ps.uniform_points_2d(0, 1, 10, 1, 1, 1) +
                          ps.uniform_points_2d(0, 0, 1, 0, 1, 10) +
                          ps.uniform_points_2d(1, 1, 1, 0, 1, 10))

solver.compile(optimizer=tf.train.GradientDescentOptimizer(0.01),
               variables=[model.weights],
               metrics=['error'])
```

Invoke `fit` method of the `solver` to do one interation of model parameters adjustment:
```python
error = solver.fit()
```
