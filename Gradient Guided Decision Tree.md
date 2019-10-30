```python
# %load I:/Machine-Learning-From-Scratch-master/gradient_boosting_decision_tree/gbd_regressor_example.py
from __future__ import division, print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import progressbar
%matplotlib inline
import sys

sys.path.append(r'I:/Machine-Learning-From-Scratch-master/')

from utils import train_test_split, standardize, to_categorical
from utils import mean_squared_error, accuracy_score, Plot
from utils.loss_functions import SquareLoss
from utils.misc import bar_widgets
from gradient_boosting_decision_tree.gbdt_model import GBDTRegressor


def main():
    print ("-- Gradient Boosting Regression --")

    # Load temperature data
    data = pd.read_csv('I:/Machine-Learning-From-Scratch-master/TempLinkoping2016.txt', sep="\t")

    time = np.atleast_2d(data["time"].as_matrix()).T
    temp = np.atleast_2d(data["temp"].as_matrix()).T

    X = time.reshape((-1, 1))               # Time. Fraction of the year [0, 1]
    X = np.insert(X, 0, values=1, axis=1)   # Insert bias term
    y = temp[:, 0]                          # Temperature. Reduce to one-dim

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    model = GBDTRegressor(learning_rate=0.03)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_pred_line = model.predict(X)

    # Color map
    cmap = plt.get_cmap('viridis')

    mse = mean_squared_error(y_test, y_pred)

    print ("Mean Squared Error:", mse)

    # Plot the results
    m1 = plt.scatter(366 * X_train[:, 1], y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(366 * X_test[:, 1], y_test, color=cmap(0.5), s=10)
    m3 = plt.scatter(366 * X_test[:, 1], y_pred, color='black', s=10)
    plt.suptitle("Regression Tree")
    plt.title("MSE: %.2f" % mse, fontsize=10)
    plt.xlabel('Day')
    plt.ylabel('Temperature in Celcius')
    plt.legend((m1, m2, m3), ("Training data", "Test data", "Prediction"), loc='lower right')
    plt.show()


if __name__ == "__main__":
    main()
```

    -- Gradient Boosting Regression --


    D:\Users\think\Anaconda3\lib\site-packages\ipykernel_launcher.py:25: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
    D:\Users\think\Anaconda3\lib\site-packages\ipykernel_launcher.py:26: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
    Training: 100% [------------------------------------------------] Time: 0:00:52


    Mean Squared Error: 9.158953842966833



![png](output_0_3.png)



```python
# %load I:/Machine-Learning-From-Scratch-master/gradient_guided_decision_tree/gg_example.py
from __future__ import division, print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import progressbar

from utils import train_test_split, standardize, to_categorical
from utils import mean_squared_error, accuracy_score, Plot
from utils.loss_functions import SquareLoss
from utils.misc import bar_widgets
from gradient_guided_decision_tree.gg_model import GGuideDTRegressor


def main():
    print ("-- Gradient Guide Regression --")

    # Load temperature data
    data = pd.read_csv('I:/Machine-Learning-From-Scratch-master/TempLinkoping2016.txt',
           sep="\t")

    time = np.atleast_2d(data["time"].as_matrix()).T
    temp = np.atleast_2d(data["temp"].as_matrix()).T

    X = time.reshape((-1, 1))               # Time. Fraction of the year [0, 1]
    X = np.insert(X, 0, values=1, axis=1)   # Insert bias term
    y = temp[:, 0]                          # Temperature. Reduce to one-dim

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    model = GGuideDTRegressor(n_estimators=250,learning_rate=0.01)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_pred_line = model.predict(X)

    # Color map
    cmap = plt.get_cmap('viridis')

    mse = mean_squared_error(y_test, y_pred)

    print ("Mean Squared Error:", mse)

    # Plot the results
    m1 = plt.scatter(366 * X_train[:, 1], y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(366 * X_test[:, 1], y_test, color=cmap(0.5), s=10)
    m3 = plt.scatter(366 * X_test[:, 1], y_pred, color='black', s=10)
    plt.suptitle("Regression Tree")
    plt.title("Test MSE: %.2f" % mse, fontsize=10)
    plt.xlabel('Day')
    plt.ylabel('Temperature in Celcius')
    plt.legend((m1, m2, m3), ("Training data", "Test data", "Prediction"), loc='lower right')
    plt.show()


if __name__ == "__main__":
    main()

```

    -- Gradient Guide Regression --


    D:\Users\think\Anaconda3\lib\site-packages\ipykernel_launcher.py:22: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
    D:\Users\think\Anaconda3\lib\site-packages\ipykernel_launcher.py:23: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
    Training: 100% [------------------------------------------------] Time: 0:00:52


    Mean Squared Error: 8.501164774559896



![png](output_1_3.png)


Test iterative decision tree in more data https://github.com/stanfordmlgroup/ngboost/tree/master/data.

- https://www.datacamp.com/community/tutorials/decision-tree-classification-python
- https://github.com/stanfordmlgroup/ngboost/tree/master/examples
- https://www.datacamp.com/community/tutorials/pandas-read-csv

We need more numerical experiment compared to boosted gradient decision tree.
The following paper give some references on the experiments.

- https://stanfordmlgroup.github.io/projects/ngboost/
- http://nicolas-hug.com/blog/around_gradient_boosting
- [Historical Gradient Boosting Machine](https://easychair.org/publications/open/pCtK)
- [Accelerated Gradient Boosting](https://arxiv.org/pdf/1803.02042.pdf)
- [Accelerating Gradient Boosting Machine](http://web.mit.edu/haihao/www/papers/AGBM.pdf)
- https://github.com/rahmacha/AdaBoost_M2_momentumNesterov

As the results shown, the GBDT is more random than gradient guide decision tree.
There are more line segment in gradient guide decision tree than gradient boost decision tree.
And gradient guide decision tree perform better than the gradient boost decision tree.


This is the direct extension of gradient descent in functional space.
And this idea is easy to generalize to any continuous optimization methods such as gradient proximal methods.
It is the first connection of continuous optimization methods and boosting (ensemble) method in iterative schemes.


____
Iterative Decision Tree | Gradient Boosting Decision Tree
:----|:----
Learn from the errors | To learn the residuals
$F^{(t+1)}\approx F^{(t)}-\nabla_{f} L(f)\mid_{f=F^{(t)}}$|$F^{(t+1)}= F^{(t)}+f_{t}$
`Fixed point` in functional /algorithmic space | `Taylor expansion` in functional /algorithmic space
Optimization update formula in target formula| Optimization procedure in update procedure
Parrallel to optimization methods | Analogous to optimization methods

Any techniques in continuous optimization can be  applied to iterative decision tree such as the distributed optimization techniques, acceleration techniques, variance reduction techniques. It is really a bridge between continuous optimization problem and decision tree algorithm by replacing the targets with update formula in optimization methods.

Compared with boosted decision tree, iterative decision tree only stores the predicted values of the last tree not all the trained trees in history.
And gradient guide (iterative) decision tree does not ensemble many trees so that it is more efficient to predict on this tree traversal.

- https://www.idug.org/p/bl/et/blogaid=646



The basic idea of gradient boosting decision tree is to fit the residuals of the last tree.
In another word, residuals plays the role of dependent variables in gradient boosted decision tree.

The [following paper](https://onlinelibrary.wiley.com/doi/abs/10.1111/1475-679X.12195) proves it is not always right.
However, gradient boosted decision tree replaces the residual with negative gradients.


- [Incorrect Inferences When Using Residuals as Dependent Variables](https://onlinelibrary.wiley.com/doi/abs/10.1111/1475-679X.12195)

Note that in iterative decision tree, we do not modify the fitting methods of the decision tree. At each step ,we fit a tree with re-defined targets,
which makes the loss function decrease.
The terminal result of iterative decision tree is only one decision tree with lowwer cost.

## Theoretical concern: convergence and generalization

Obviously, there is no room to improve when the functional gradient is 0s, i.e., $\nabla_{f} L(f)\mid_{f=F^{(t)}}=0$ like in gradient descent.
The theoretical concern is its convergence and generalization ability.
Like stochastic gradient descent, gradient guided decision tree does not perform the gradient descent exactly
because the decision tree cannot fit the target $F^{(t)}-\alpha_t\nabla_{f} L(f)\mid_{f=F^{(t)}}$ without any error.
In this sense, it is similar to stochastic gradient descent.

- https://lizhongresearch.miraheze.org/wiki/%E7%90%86%E8%A7%A3%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C
- http://downloads.hindawi.com/journals/jece/2015/835357.pdf
- https://enesdilber.github.io/GiDL.pdf



See [A collection of research papers on decision, classification and regression trees with implementations.](https://github.com/benedekrozemberczki/awesome-decision-tree-papers) for more information on decision tree .

## Bayesian Decision Tree and Regularization of Iterative Decision Tree

`Decision Tree` is always considered as non-parametric regression method in statistics
and step function in mathematical analysis(or simple function in measure theory).

In general, the cost function of supervised machine learn consists of two parts:

* the sum of loss function of each sample $\sum_{n=1}^{N}\ell(f(x_n), y_n)$;
* the regularization term to control the model denoted as $\Omega(f)$.

Here $\Omega$ is a functional mapping the model $f$ to some nonnegative real number, i.e.,
$$\Omega:\mathcal{F}\mapsto \mathbb{R}^{+}$$
where $\mathcal F$ is the model space containing some types of functions.


- https://repository.upenn.edu/statistics_papers/370/
- http://rob-mcculloch.org/code/pbart.pdf
- https://astro.temple.edu/~msobel/
- http://www-stat.wharton.upenn.edu/~steele/
- http://rob-mcculloch.org/
- https://faculty.chicagobooth.edu/richard.hahn
- https://manduca.la.asu.edu/~prhahn/
- https://github.com/JakeColtman/bartpy
- https://jakecoltman.github.io/bartpy/
- https://faculty.chicagobooth.edu/veronika.rockova/

In iterative decision tree, the problem is how to compute the gradient of
regularization terms with respect to the decision tree $\frac{\partial }{\partial f}\Omega(F)$.

What is more, the regularization term is not always differentiable.
The good news is that it is possible to compute these parts separately.

`Our problem is to find a new decision tree that fit a new targets and statisfy some constraints of the decision tree itself meanwhile.`
In mathematics,
$$\arg\min_{f\in\mathcal{F}}L(f)(=\sum_{n=1}^{N}\ell(f(x_n), y_n)+\Omega(f))$$


``Alternate minimization``: (1) fit a tree with the targets $f^{(t)}(x_n-\alpha_t\{\nabla_{f(x)} \ell(f(x_n), y_n)\mid_{f(x)=f^{(t)}(x_n)}\}$ and output the new tree $f^{(t+1)}$;
(2) minimize the regularization term $\Omega(f)$ as post-pruning.

Like xGBoost, we can fit a new tree with tree construction method of `extreme gradient boosting tree`.

- http://datascience.la/xgboost-workshop-and-meetup-talk-with-tianqi-chen/
- http://dmlc.cs.washington.edu/data/pdf/XGBoostArxiv.pdf
- https://xgboost.readthedocs.io/en/latest/tutorials/model.html

### Newton Guided Decision Tree

The following formula is Newton-type update formula：
$$f_{t+1}\approx f_{t}-\frac{\partial L(f_t)}{\partial^2 L(f_t)}.$$

If we fit a new tree with such new targets $\{f_{t}(x_n)-\frac{\partial L(f_t(x_n), y_n)}{\partial^2 L(f_t(x_n), y_n)}\mid n=1,2,\cdots, N\}$,
it is supposed to compare with `extreme gradient boosting tree`. This is so-called `Newton guided decision tree `.

- http://mariofilho.com/can-gradient-boosting-learn-simple-arithmetic/
- https://github.com/szilard/benchm-ml
- https://zhuanlan.zhihu.com/p/62670784

### Decision Tree and Binary Network

[Binary neural networks are networks with binary weights and activations at run time. At training time these weights and activations are used for computing gradients; however, the gradients and true weights are stored in full precision. This procedure allows us to effectively train a network on systems with fewer resources.](https://software.intel.com/en-us/articles/binary-neural-networks)

This network has the following layers:
![Layer map](https://software.intel.com/sites/default/files/managed/c0/e0/webops10048-fig4-network-layers.png)


* Fully connected (128)
* Ramp - rectified linear unit (ReLU) activation function
* Binarize activations
* Fully connected (128)
* Ramp - ReLU activation function
* Binarize activations
* Fully connected (10)
* Sigmoid activation function

- https://blog.csdn.net/stdcoutzyx/article/details/50926174
- https://duanyzhi.github.io/Binary-Network/
- https://software.intel.com/en-us/articles/binary-neural-networks
- https://github.com/MatthieuCourbariaux/BinaryConnect

Decision tree seems far different from deep neural network.
There is no back-propagation/loss reduction methods in decision tree.
And it heavily depends on the input samples.

In this section, we will show some similarities of decision tree and neural network in the sense of computational graph.

First we observe that the leaf value is the average of the instances' labels in regression tree.
It is similar to average convolution in deep convolution network.
Second, decision tree is hierarchical model regarded as directed acyclic graph(DAG) in computational graph sense.
In fact, it is pipeline where the outputs of the middle nodes only depend on its higher level.

Third, leaf nodes output the results. Thus even every sample is in different level of decision tree,
each sample only is mapped into a leaf node.

- [QuickScorer: A Fast Algorithm to Rank Documents with Additive Ensembles of Regression Trees](https://dl.acm.org/citation.cfm?doid=2766462.2767733)
