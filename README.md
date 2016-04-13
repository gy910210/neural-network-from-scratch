# Implementing Multiple Layer Neural Network from Scratch
This post is inspired by <http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch>.

In this post, we will implement a multiple layer neural network from scratch. You can regard the number of layers and dimension of each layer as parameter. For example, `[2, 3, 2]` represents inputs with 2 dimension, one hidden layer with 3 dimension and output with 2 dimension (binary classification) (using softmax as output).

We won’t derive all the math that’s required, but I will try to give an intuitive explanation of what we are doing. I will also point to resources for you read up on the details.

## Generating a dataset
Let’s start by generating a dataset we can play with. Fortunately, [scikit-learn](http://scikit-learn.org/) has some useful dataset generators, so we don’t need to write the code ourselves. We will go with the [make_moons](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html) function.

```python
# Generate a dataset and plot it
np.random.seed(0)
X, y = sklearn.datasets.make_moons(200, noise=0.20)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
```
![](https://github.com/pangolulu/neural-network-from-scratch/raw/master/figures/nn-from-scratch-dataset.png)

The dataset we generated has two classes, plotted as red and blue points. Our goal is to train a Machine Learning classifier that predicts the correct class given the x- and y- coordinates. Note that the data is not linearly separable, we can’t draw a straight line that separates the two classes. This means that linear classifiers, such as Logistic Regression, won’t be able to fit the data unless you hand-engineer non-linear features (such as polynomials) that work well for the given dataset.

In fact, that’s one of the major advantages of Neural Networks. You don’t need to worry about feature engineering. The hidden layer of a neural network will learn features for you.

## Neural Network
### Neural Network Architecture
You can read this tutorial (<http://cs231n.github.io/neural-networks-1/>) to learn the basic concepts of neural network. Like activation functions, feed-forward computation and so on.

Because we want our network to output probabilities the activation function for the output layer will be the [softmax](https://en.wikipedia.org/wiki/Softmax_function), which is simply a way to convert raw scores to probabilities. If you’re familiar with the logistic function you can think of softmax as its generalization to multiple classes.

When you choose softmax as output, you can use [cross-entropy loss](https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_error_function_and_logistic_regression) (also known as negative log likelihood) as loss function. More about Loss Function can be find in <http://cs231n.github.io/neural-networks-2/#losses>.
### Learning the Parameters
Learning the parameters for our network means finding parameters (such as (W_1, b_1, W_2, b_2)) that minimize the error on our training data (loss function).

We can use [gradient descent](http://cs231n.github.io/optimization-1/) to find the minimum and I will implement the most vanilla version of gradient descent, also called batch gradient descent with a fixed learning rate. Variations such as SGD (stochastic gradient descent) or minibatch gradient descent typically perform better in practice. So if you are serious you’ll want to use one of these, and ideally you would also [decay the learning rate over time](http://cs231n.github.io/neural-networks-3/#anneal).

The key of gradient descent method is how to calculate the gradient of loss function by the parameters. One approach is called [Back Propagation](https://en.wikipedia.org/wiki/Backpropagation). You can learn it more from <http://colah.github.io/posts/2015-08-Backprop/> and <http://cs231n.github.io/optimization-2/>.

### Implementation
We start by given the computation graph of neural network.
![](https://github.com/pangolulu/neural-network-from-scratch/raw/master/figures/computation-graph.png)

In the computation graph, you can see that it contains three components (`gate`, `layer` and `output`), there is two kinds of gate (`multiply` and `add`), and you can use `tanh` layer and `softmax` output.

`gate`, `layer` and `output` can all be seen as operation unit of computation graph, so they will implement the inner derivatives of their inputs (we call it `backward`), and use chain rule according to the computation graph.

**`gate.py`**
```python
import numpy as np

class MultiplyGate:
    def forward(self,W, X):
        return np.dot(X, W)

    def backward(self, W, X, dZ):
        dW = np.dot(np.transpose(X), dZ)
        dX = np.dot(dZ, np.transpose(W))
        return dW, dX

class AddGate:
    def forward(self, X, b):
        return X + b

    def backward(self, X, b, dZ):
        dX = dZ * np.ones_like(X)
        db = np.dot(np.ones((1, dZ.shape[0]), dtype=np.float64), dZ)
        return db, dX
```

**`layer.py`**
```python
import numpy as np

class Sigmoid:
    def forward(self, X):
        return 1.0 / (1.0 + np.exp(-X))

    def backward(self, X, top_diff):
        output = self.forward(X)
        return (1.0 - output) * output * top_diff

class Tanh:
    def forward(self, X):
        return np.tanh(X)

    def backward(self, X, top_diff):
        output = self.forward(X)
        return (1.0 - np.square(output)) * top_diff
```
**`output.py`**
```python
import numpy as np

class Softmax:
    def predict(self, X):
        exp_scores = np.exp(X)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def loss(self, X, y):
        num_examples = X.shape[0]
        probs = self.predict(X)
        corect_logprobs = -np.log(probs[range(num_examples), y])
        data_loss = np.sum(corect_logprobs)
        return 1./num_examples * data_loss

    def diff(self, X, y):
        num_examples = X.shape[0]
        probs = self.predict(X)
        probs[range(num_examples), y] -= 1
        return probs
```

We can implement out neural network by a class `Model` and initialize the parameters in the `__init__` function. You can pass the parameter `layers_dim = [2, 3, 2]`, which represents inputs with 2 dimension, one hidden layer with 3 dimension and output with 2 dimension
```python
class Model:
    def __init__(self, layers_dim):
        self.b = []
        self.W = []
        for i in range(len(layers_dim)-1):
            self.W.append(np.random.randn(layers_dim[i], layers_dim[i+1]) / np.sqrt(layers_dim[i]))
            self.b.append(np.random.randn(layers_dim[i+1]).reshape(1, layers_dim[i+1]))
```

First let’s implement the loss function we defined above. It is just a forward propagation computation of out neural network. We use this to evaluate how well our model is doing:
```python
def calculate_loss(self, X, y):
    mulGate = MultiplyGate()
    addGate = AddGate()
    layer = Tanh()
    softmaxOutput = Softmax()

    input = X
    for i in range(len(self.W)):
        mul = mulGate.forward(self.W[i], input)
        add = addGate.forward(mul, self.b[i])
        input = layer.forward(add)

    return softmaxOutput.loss(input, y)
```
We also implement a helper function to calculate the output of the network. It does forward propagation as defined above and returns the class with the highest probability.
```python
def predict(self, X):
    mulGate = MultiplyGate()
    addGate = AddGate()
    layer = Tanh()
    softmaxOutput = Softmax()

    input = X
    for i in range(len(self.W)):
        mul = mulGate.forward(self.W[i], input)
        add = addGate.forward(mul, self.b[i])
        input = layer.forward(add)

    probs = softmaxOutput.predict(input)
    return np.argmax(probs, axis=1)
```
Finally, here comes the function to train our Neural Network. It implements batch gradient descent using the backpropagation algorithms we have learned above.

```python
def train(self, X, y, num_passes=20000, epsilon=0.01, reg_lambda=0.01, print_loss=False):
    mulGate = MultiplyGate()
    addGate = AddGate()
    layer = Tanh()
    softmaxOutput = Softmax()

    for epoch in range(num_passes):
        # Forward propagation
        input = X
        forward = [(None, None, input)]
        for i in range(len(self.W)):
            mul = mulGate.forward(self.W[i], input)
            add = addGate.forward(mul, self.b[i])
            input = layer.forward(add)
            forward.append((mul, add, input))

        # Back propagation
        dtanh = softmaxOutput.diff(forward[len(forward)-1][2], y)
        for i in range(len(forward)-1, 0, -1):
            dadd = layer.backward(forward[i][1], dtanh)
            db, dmul = addGate.backward(forward[i][0], self.b[i-1], dadd)
            dW, dtanh = mulGate.backward(self.W[i-1], forward[i-1][2], dmul)
            # Add regularization terms (b1 and b2 don't have regularization terms)
            dW += reg_lambda * self.W[i-1]
            # Gradient descent parameter update
            self.b[i-1] += -epsilon * db
            self.W[i-1] += -epsilon * dW

        if print_loss and epoch % 1000 == 0:
            print("Loss after iteration %i: %f" %(epoch, self.calculate_loss(X, y)))
```
### A network with a hidden layer of size 3
Let’s see what happens if we train a network with a hidden layer size of 3.
```python
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import mlnn
from utils import plot_decision_boundary

# Generate a dataset and plot it
np.random.seed(0)
X, y = sklearn.datasets.make_moons(200, noise=0.20)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
plt.show()

layers_dim = [2, 3, 2]

model = mlnn.Model(layers_dim)
model.train(X, y, num_passes=20000, epsilon=0.01, reg_lambda=0.01, print_loss=True)

# Plot the decision boundary
plot_decision_boundary(lambda x: model.predict(x), X, y)
plt.title("Decision Boundary for hidden layer size 3")
plt.show()
```
![](https://github.com/pangolulu/neural-network-from-scratch/raw/master/figures/nn-from-scratch-h3.png)

This looks pretty good. Our neural networks was able to find a decision boundary that successfully separates the classes.

The `plot_decision_boundary` function is referenced by <http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch>.
```python
import matplotlib.pyplot as plt
import numpy as np

# Helper function to plot a decision boundary.
def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
```
## Further more
1. Instead of batch gradient descent, use minibatch gradient to train the network. Minibatch gradient descent typically performs better in practice ([more info](http://cs231n.github.io/optimization-1/#gd)).
2. We used a fixed learning rate `epsilon` for gradient descent. Implement an annealing schedule for the gradient descent learning rate ([more info](http://cs231n.github.io/neural-networks-3/#anneal)).
3. We used a `tanh` activation function for our hidden layer. Experiment with other activation functions ([more info](http://cs231n.github.io/neural-networks-1/#actfun)).
4. Extend the network from two to three classes. You will need to generate an appropriate dataset for this.
5. Try some other Parameter updates method, like `Momentum update`, `Nesterov momentum`, `Adagrad`, `RMSprop` and `Adam`([more info](http://cs231n.github.io/neural-networks-3/#update)).
6. Some other tricks of training neural network can be find <http://cs231n.github.io/neural-networks-2> and <http://cs231n.github.io/neural-networks-3>, like `dropout reglarization`, `batch normazation`, `Gradient checks` and `Model Ensembles`.

## Some useful resources
1. <http://cs231n.github.io/>
2. <http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch>
3. <http://colah.github.io/posts/2015-08-Backprop/>
