import numpy as np

y_predictions = np.zeros(100)
five_fold_average_accuracy  = 0.0
tuned_stepsize = 0.0
#====================================================================
def sigm(z):
    """
    Computes the sigmoid function

    :type z: float
    :rtype: float
    """
    return 1/(1+np.exp(-z))

#====================================================================
def compute_grad(w, x, y):
    """
    Computes gradient of LL for logistic regression

    :type w: 1D np array of weights
    :type x: 2D np array of features where len(w) == len(x[0])
    :type y: 1D np array of labels where len(x) == len(y)
    :rtype: 1D numpy array
    """
    gradient = np.zeros(len(w))
    for i in range(x.shape[0]):
        xi = x[i]
        g = sigm(w.dot(xi))
        gradient = gradient + xi *(y[i] - g)
    return gradient
#====================================================================
def gd_single_epoch(w, x, y, step):
    """
    Updates the weight vector by processing the entire training data once

    :type w: 1D numpy array of weights
    :type x: 2D numpy array of features where len(w) == len(x[0])
    :type y: 1D numpy array of labels where len(x) == len(y)
    :rtype: 1D numpy array of weights
    """
    weight_next = np.array(len(w))
    weight_next = w + step*(compute_grad(w, x, y))
    return weight_next
w_single_epoch = gd_single_epoch(np.zeros(len(x_train[0])), x_train, y_train, default_stepsize)
# print(w_single_epoch)

#====================================================================
def gd(x, y, stepsize):
    """
    Iteratively optimizes the objective function by first
    initializing the weight vector with zeros and then
    iteratively updates the weight vector by going through
    the trianing data num_epoch_for_train(global var) times

    :type x: 2D numpy array of features where len(w) == len(x[0])
    :type y: 1D numpy array of labels where len(x) == len(y)
    :type stepsize: float
    :rtype: 1D numpy array of weights
    """
    wk = np.zeros(len(x_train[0]))
    for i in range(num_epoch_for_train):
        wk = gd_single_epoch(wk, x, y, stepsize)
    return wk

w_optimized = gd(x_train, y_train, default_stepsize)

#====================================================================
def predict(w, x):
    """
    Makes a binary decision {0,1} based the weight vector
    and the input features

    :type w: 1D numpy array of weights
    :type x: 1D numpy array of features of a single data point
    :rtype: integer {0,1}
    """
    y = 0
    if sigm(w.dot(x)) > 0.5:
        y = 1
    return y

y_predictions = np.fromiter((predict(w_optimized, xi) for xi in x_test), x_test.dtype)

#====================================================================
def accuracy(w, x, y):
    """
    Calculates the proportion of correctly predicted results to the total

    :type w: 1D numpy array of weights
    :type x: 2D numpy array of features where len(w) == len(x[0])
    :type y: 1D numpy array of labels where len(x) == len(y)
    :rtype: float as a proportion of correct labels to the total
    """
    y_predicted = np.zeros(len(y))
    for i in range(len(y)):
        y_predicted[i] = predict(w, x[i])
    acc = sum(y_predicted == y)/len(y)
    return acc

# print("presiction acc = ", accuracy(w_optimized, x_test, y_test))

#====================================================================
def five_fold_cross_validation_avg_accuracy(x, y, stepsize):
    """
    Measures the 5 fold cross validation average accuracy
    Partition the data into five equal size sets like
    |-----|-----|-----|-----|
    For all 5 choose 1 permutations, train on 4, test on 1.

    Compute the average accuracy using the accuracy function
    you wrote.

    :type x: 2D numpy array of features where len(w) == len(x[0])
    :type y: 1D numpy array of labels where len(x) == len(y)
    :type stepsize: float
    :rtype: float as average accuracy across the 5 folds
    """
#======TEST 1====================
    wi = gd(x[60:300], y[60:300], stepsize)
    acc1 = accuracy(wi, x[0:60], y[0:60])
    # print("acc1",acc1)
#======TEST 2====================
    wi = gd(np.concatenate([x[0:60],x[120:300]]), np.concatenate([y[0:60],y[120:300]]), stepsize)
    acc2 = accuracy(wi, x[60:120], y[60:120])
    # print("acc2",acc2)
#======TEST 3====================
    wi = gd(np.concatenate([x[0:120],x[180:300]]), np.concatenate([y[0:120],y[180:300]]), stepsize)
    acc3 = accuracy(wi, x[120:180], y[120:180])
    # print("acc3",acc3)
#======TEST 4====================
    wi = gd(np.concatenate([x[0:180],x[240:300]]), np.concatenate([y[0:180],y[240:300]]), stepsize)
    acc4 = accuracy(wi, x[180:240], y[180:240])
    # print("acc4",acc4)
#======TEST 5====================
    wi = gd(x[0:240], y[0:240], stepsize)
    acc5 = accuracy(wi, x[240:300], y[240:300])
    # print("acc5",acc5)
    return (acc1+acc2+acc3+acc4+acc5)/5


five_fold_average_accuracy = five_fold_cross_validation_avg_accuracy(x_train, y_train, default_stepsize)
def tune(x, y):
    """
    Optimizes the stepsize by calculating five_fold_cross_validation_avg_accuracy
    with 10 different stepsizes from 0.001, 0.002,...,0.01 in intervals of 0.001 and
    output the stepsize with the highest accuracy

    For comparison:
    If two accuracies are equal, pick the lower stepsize.

    NOTE: For best practices, we should be using Nested Cross-Validation for
    hyper-parameter search. Without Nested Cross-Validation, we bias the model to the
    data. We will not implement nested cross-validation for now. You can experiment with
    it yourself.
    See: http://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html

    :type x: 2D numpy array of features where len(w) == len(x[0])
    :type y: 1D numpy array of labels where len(x) == len(y)
    :rtype: float as best stepsize
    """
    acc_max = 0
    optimal_step = 0.001
    steps = np.arange(0.001,0.011,0.001)
    #print(steps)
    for step_i in steps:
        #print(step_i)
        acc_i = five_fold_cross_validation_avg_accuracy(x_train, y_train, step_i)
        #print(acc_i)
        if acc_i > acc_max:
            optimal_step = step_i
            acc_max = acc_i
    #print(optimal_step)
    return optimal_step
tuned_stepsize = tune(x_train, y_train)
