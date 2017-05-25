# STAT558
This code implements two algorithms: 

1. gradient algorithm, and
2. fast gradient algorithm.

to solve l2-regularized logistic regression problem.

Graddescent and fastgradalgo algorithm optimize the objective value of logisitic regression. And plot the curve of the objective value for both algorithms. And plot the misclassfication error for both algorithm vesus the iteration counter t.

See the example.py file for an example of how to use them and the comments in Gradient_Descent_vs_Fast_Gradient_Descent.py for all of the possible input options.

Briefly, you can load the file using
```
import Gradient_Descent_vs_Fast_Gradient_Descent
```

Specify your data souce
simple simulated dataset
```


```
real-world dataset
```
https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/spam.data', sep=' ', header=None
```
and then run
```
cr = src.cubic_reg.CubicRegularization(x0, f=f, gradient=grad, hessian=hess, conv_tol=1e-4)
x_opt, intermediate_points, n_iter, flag = cr.cubic_reg()
```

There are many other options you can specify and parameters you can control.
