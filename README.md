# STAT558
Gradient algorithm, Fast gradeitn algorithm and scikit learn on Binary Logistic Classification.

This code implements two algorithms: 

1. gradient algorithm, and
2. fast gradient algorithm.

to solve l2-regularized logistic classification problem.

Graddescent and fastgradalgo algorithm optimize the objective value of logisitic regression. And plot the curve of the objective value for both algorithms. And plot the misclassfication error for both algorithm vesus the iteration counter t.

See the demo.ipynb file for examples of how to use them all of the possible input options. There are three options:
```
0, Real World Data by Algorithm: 
   Calculate optimal beta values from gradient and fast gradient algorithm. And plot the curve of the objective value for both algorithms versus the iteration. And plot the misclassification error for both algorithm versus iterations.
   Data: Spam data from https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/spam.data
1, Simple Data by Algorithm: 
   Calculate optimal beta values from gradient and fast gradient algorithm. And plot the curve of the objective value for both algorithms versus the iteration. And plot the misclassification error for both algorithm versus iterations.
   Data: scikt learn built-in data iris. Can be import directed by using from sklearn.datasets import load_iris.
2, Comparison with Scikit-Learn (Real World Data)]:
   Compare optimal beta values from fast gradient algorithm and scikit learn using the same lambda value. And compare the corresponding objective value from fast gradient algorithm and scikit learn.
```

Briefly, you can run the file by choosing an option, and following files will be run:
```
  option = int(input("Enter an option [0 = Real World Data by Algorithm, 1 = Simple Data by Algorithm, 2 = Comparison with Scikit-Learn (Real World Data)]: "))
  if option == 0:
    %run Algorithm_RealData.py
  elif option == 1:
    %run Algorithm_SimpleData.py
  elif option ==2:
    %run Comparison.py
```

Data Souce:
real-world dataset
```
https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/spam.data
```
simpe dataset, run
```
sklearn.datasets import load_iris
iris = load_iris()
```

There are many other options you can specify and parameters you can control.
