import numpy as np
import matplotlib.pyplot as plt
import copy
def sigmoid(z):
    """
    sigmoid function
    :param ndarray z
    """
    return 1.0/(1.0+np.exp(-z))

def prime_sigmoid(z):
    """
    sigmoid function
    :param ndarray z
    """
    return np.exp(-z)/((1+np.exp(-z))**2)

def predict(x, w, b):
    """
    Forward prediction of neural network
    :param ndarray x: num_feature x 1 numpy array
    :param list w: follows the format of "weights" declared below
    :param list b: follows the format of "bias" declared below
    :rtype int: label index, starting from 1
    """
    #d = x.shape[1]
    n = x.shape[0]
    #zh = bh + x1.dot(w_ih)
    # print("x.shape", x.shape) = 13
    # weights = [np.ones((num_hidden_nodes, num_feature)), np.ones((num_label, num_hidden_nodes))]
    # bias = [np.zeros((num_hidden_nodes, 1)), np.zeros((num_label, 1))]
    wh = w[0]
    wk = w[1]
    bh = b[0]
    bk = b[1]
    zh = bh + np.dot(wh, x)
    gh = sigmoid(zh)
    #
    zk = bk + np.dot(wk, gh)
    gk = sigmoid(zk)
    # gk = ak
    # debug infor
    if gk.shape!= (3,1):
        print("x.shape", x.shape)
        print("gk.shape",gk.shape)
        print("wh = w[0].shape,", wh.shape)
        print("wk = w[1].shape,", wk.shape)
        print("bh = b[0].shape,", bh.shape)
        print("bk = b[1].shape,", bk.shape)
        print("zh = bh + np.dot(wh, x) .shapr", zh.shape)
        print("gh = sigmoid(zh) shape", gh.shape)
        print("zk = bk + np.dot(wk, gh) shape ", zk.shape)
        print("gk = sigmoid(zk)")
    return np.argmax(gk)+1

def accuracy(testing_data, testing_label, w, b):
    """
    Return the accuracy(0 to 1) of the model w, b on testing data
    :param ndarray testing_data: num_data x num_feature numpy array
    :param ndarray testing_label: num_data x 1 numpy array
    :param list w: follows the format of "weights" declared below
    :param list b: follows the format of "bias" declared below
    :rtype float: accuracy(0 to 1)
    """
    acc_num = sum(predict(testing_data[i].reshape((testing_data.shape[1],1)),w,b)==testing_label[i] for i in range(testing_data.shape[0]))
    return acc_num/testing_data.shape[0]

def gradient(x, y, w, b):
    """
    Compute gradient using backpropagation
    :param ndarray x: num_feature x 1 numpy array
    :param ndarray y: num_label x 1 numpy array
    :rtype tuple: A tuple contains the delta/gradient of weights and bias (dw, db)
                dw and db should have same format as w and b correspondingly
    """
    # weights = [np.ones((num_hidden_nodes, num_feature)), np.ones((num_label, num_hidden_nodes))]
    # bias = [np.zeros((num_hidden_nodes, 1)), np.zeros((num_label, 1))]
    # py is predicted_y
    wh = w[0]
    wk = w[1]
    bh = b[0]
    bk = b[1]
    zh = bh + np.dot(wh, x)
    gh = sigmoid(zh)
    zk = bk + np.dot(wk, gh)
    gk = sigmoid(zk)
    prim_gzk = prime_sigmoid(zk)
    prim_gzh = prime_sigmoid(zh)
    #print("gk",gk)
    #print("y", y)
    #print("prim_gzk.shape", prim_gzk.shape)
    #print("gh.shape",gh.shape)
    #print("abs(gk - y).shape)", abs(gk - y).shape)
    #print("(gk - y).shape", (gk - y).shape)
    # output layer
    dE_dwk = np.matmul((gk - y)*prim_gzk, gh.T)
    dE_dbk = ((gk - y)*prim_gzk).reshape(bk.shape)
    # hidden layer
    sum_dkwjk = np.sum((gk - y)*prim_gzk*wk, axis = 0).reshape(zh.shape)
    dE_dwh = sum_dkwjk * prim_gzh * x.T
    dE_dbh = (sum_dkwjk * prim_gzh).reshape(bh.shape)
    return ([dE_dwh, dE_dwk], [dE_dbh, dE_dbk])

def single_epoch(w, b, training_data, training_label, eta, num_label):
    """
    Compute one epoch of batch gradient descent
    :param list w: follows the format of "weights" declared below
    :param list b: follows the format of "bias" declared below
    :param ndarray training_data: num_data x num_feature numpy array
    :param ndarray training_label: num_data x 1 numpy array
    :param float eta: step size
    :param int num_label: number of labels
    :rtype tuple: A tuple contains the updated weights and bias (w, b)
                w and b should have same format as they are pased in
    """
    # weight = weight - eta*(dE-dw)
    sum_d_wh = np.zeros(w[0].shape)
    sum_d_wk = np.zeros(w[1].shape)
    sum_d_bh = np.zeros(b[0].shape)
    sum_d_bk = np.zeros(b[1].shape)
    for i in range (training_data.shape[0]):
        # one hot vector conversion
        y = np.zeros(num_label)
        y[training_label[i] - 1] += 1
        y = y.reshape((num_label,1))
        dw, db = gradient(training_data[i].reshape((training_data.shape[1],1)), y, w, b)
        sum_d_wh += dw[0]
        sum_d_wk += dw[1]
        sum_d_bh += db[0]
        sum_d_bk += db[1]
    w[0] -= eta*sum_d_wh/training_data.shape[0]
    w[1] -= eta*sum_d_wk/training_data.shape[0]
    b[0] -= eta*sum_d_bh/training_data.shape[0]
    b[1] -= eta*sum_d_bk/training_data.shape[0]
    return (w,b)
    
def batch_gradient_descent(w, b, training_data, training_label, eta, num_label, num_epochs = 200):
    """
    :param list w: follows the format of "weights" declared below
    :param list b: follows the format of "bias" declared below
    :param ndarray training_data: num_data x num_feature numpy array
    :param ndarray training_label: num_data x 1 numpy array
    :param float eta: step size
    :param int num_label: number of labels
    :rtype tuple: A tuple contains the updated weights and bias (w, b)
                w and b should have same format as they are pased in
    """
    rw = copy.deepcopy(w)
    rb = copy.deepcopy(b)
    acc = np.zeros(num_epochs)
    for i in range(num_epochs):
        #if i%10 = 0:
            #test_acc = accuracy(testing_data, testing_label, w, b)
            #train_acc = accuracy((training_data, training_label, w, b))
            #print 'start epoch {}, train acc {} test acc {}'.format(i, train_acc, test_acc)
        rw, rb = single_epoch(rw, rb, training_data, training_label, eta, num_label)
        acc[i] = accuracy(training_data, training_label, rw, rb)
    if eta in step_sizes:
        plt.figure()
        plt.plot(acc)
        plt.ylabel('Accuracy')
        plt.xlabel('xth_epoch')
        title = "Step size (eta) = " + str(eta)
        plt.title(title)
        plt.show()
    return (rw, rb)
num_label = 3
num_feature = len(training_data[0])
num_hidden_nodes = 50 #50 is not the best parameter, but we fix it here
step_sizes = [0.3,3,10]
#REFER the dimension and format here
#sizes =[num_feature, num_hidden_nodes, num_label]

##init_weights = [shape_of_matrix(num_hidden_nodes, num_feature) for i = 0]
##init_weights = [shape_of_matrix(num_label, num_hidden_nodes) for i = 1]
#init_bias = [shape_of_matrix(num_hidden_nodes, 1) for i = 0]
#init_bias = [shape_of_matrix(num_label, 1) for i = 1]

#w_shape = ((num_hidden_nodes, num_feature),(num_label, num_hidden_nodes))
weights = [np.ones((num_hidden_nodes, num_feature)), np.ones((num_label, num_hidden_nodes))]
bias = [np.zeros((num_hidden_nodes, 1)), np.zeros((num_label, 1))]
#init_weights = [prng.randn(sizes[i+1], sizes[i]) for i in range(len(sizes)-1)]
#init_bias = [prng.randn(sizes[i+1],1) for i in range(len(sizes)-1)]
#END don't touch

#ATTENTION:
# If you are going to call batch_gradient_descent multiple times
# DO MAKE A DEEP COPY OF init_weights AND init_bias BEFORE CALLING!
# Or MAKE A DEEP COPY when use them in batch_gradient_descent
weights_a, bias_a = batch_gradient_descent(init_weights, init_bias, training_data, training_label, step_sizes[0], 3)
weights_b, bias_b = batch_gradient_descent(init_weights, init_bias, training_data, training_label, step_sizes[1], 3)
weights_c, bias_c = batch_gradient_descent(init_weights, init_bias, training_data, training_label, step_sizes[2], 3)
weights = weights_b
bias = bias_b
print("accuracy of parameters (w,b) with eta (",step_sizes[0],") in training data is ", accuracy(training_data, training_label, weights_a, bias_a))
print("accuracy of parameters (w,b) with eta (",step_sizes[0],") in testing data is ", accuracy(testing_data, testing_label, weights_a, bias_a))
print("")
print("accuracy of parameters (w,b) with eta (",step_sizes[1],") in training data is ", accuracy(training_data, training_label, weights_b, bias_b))
print("accuracy of parameters (w,b) with eta (",step_sizes[1],") in testing data is ", accuracy(testing_data, testing_label, weights_b, bias_b))
print("")
print("accuracy of parameters (w,b) with eta (",step_sizes[2],") in training data is ", accuracy(training_data, training_label, weights_c, bias_c))
print("accuracy of parameters (w,b) with eta (",step_sizes[2],") in testing data is ", accuracy(testing_data, testing_label, weights_c, bias_c))

print("From the accuracy rate we can clear see that the parameters with step size eta = 3 has the most optimal accuracy rate.")
print("From the graph, we can see that the second graph is more stable and for epoch = 200,","\n","the accuracy is always in an increasing trend as we take more steps.")
print("But the other two all have problems: ")
print("When we have step size as 0.3, it is too small that we do not get any accuracy improve after 50 epoch.","\n","And it’s highest accuracy is very low.")
print("When we have step size as 10, it is too big that we could land to the optimal parameters.","\n","The accuracy falls after around 50 epochs. ")
print("Therefore we choice to set step size = 3, that after 200 epoch gives us the parameters that have the highest accuracy.")

#print accuracy(training_data, training_label, weights, bias)
#print accuracy(testing_data, testing_label, weights, bias)
