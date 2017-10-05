import numpy as np
import matplotlib.pyplot as plt

# Step 1: Create the plots
"""
Ploting For each of the x[i] input features in x.
Use matplotlib to create your plots.
We should see that certain features are more closely correlated with y than others.
We will use this information later on.
"""

data_x = golf_data[:, :-1]
data_y = golf_data[:, 11]
#-------------------------------
def plot_all_features(data_x, data_y):
    for i in range (data_x.shape[1]):
        plt.figure()
        title = "Fiture x["+str(i)+"]"
        plt.title(title)
        plt.scatter(data_x[:,i], data_y)
        plt.show()
plot_all_features(data_x, data_y)


# Step 2: Define the linear regression function using gradient descent
def gradient(data, weights):
    """
    Computes the gradient of the residual sum-of-squares (RSS) function
    for the given dataset and current weight values

    :param numpy.ndarray data: A (n, m) numpy array,
    where n = number of examples and m=(features + 1).
    The last column is the label y
    for that example:
    :param numpy.ndarray weights: A (m,) numpy array,
    where weights[m-1] is the bias, and weights[i] corresponds to the
    weight for feature i in data
    :returns A (m,) numpy array, equaling the gradient of RSS at this point.
    :rtype: numpy.ndarray

    """
    n = len(data)
    m = data.shape[1]
    weight_next = np.zeros(m)
    y = data[:, m-1]
    x = np.zeros((n,m))
    x[:,:(m-1)] = data[:,:(m-1)]
    x[:,(m-1)] = 1
    # RSS = ||y - Xw||^2_2
    # Gradient(RSS) = 2xTxw-2xTy
    # weight_next = 2*(x.T @ x @ weights - x.T @ y)
    weight_next = 2*(np.matmul(np.matmul(x.T , x) , weights) - np.matmul(x.T , y))
    weight_next = (weight_next/n)
    return weight_next


def gradient_descent(data, alpha, iterations):
    """
    Performs gradient descent using the supplied data, learning rate, and number of iterations. 
    Start with weights =
    the zero vector.

    :param numpy.ndarray data: A (n, m) numpy array, where n = number of examples and m=(features + 1). The last column
    is the label y for that example
    :param float alpha: A real value corresponding to the step size for each descent step
    :param int iterations: The total number of iterations (steps) to take
    :returns A (m,) numpy array of the final weight vector after the gradient descent process
    :rtype: numpy.ndarray
    """
    # w_t+1 = wt - alpha * hradient(w)
    w_final = np.zeros(data.shape[1])
    for i in range (iterations):
        w_final = w_final - alpha*gradient(data, w_final)
    return w_final


# Step 3: Standardize the features (but not the labels)
golf_data_standardized = None  # Implement me!
def normailze_data(data):
    n_data = np.zeros((data.shape[0], data.shape[1]))
    n_data = np.copy(data)
    for i in range(data.shape[1]-1): #last colum do not normalize
        mul = np.average(n_data[:,i])
        std = np.sqrt(sum((n_data[:,i] - mul)**2)/len(data))
        n_data[:,i] = (n_data[:,i] - mul)/std
    #print(n_data)
    return n_data



golf_data_standardized = normailze_data(golf_data)
# Step 4: Implement Forward Selection
def forward_selection(data, max_var):
    # default varlue:
    alpha = 0.1
    iterations = 200
    # Useful latter:
    lenth = data.shape[0]
    width = data.shape[1]
    # y is the last colum in the data
    y = data[:, width-1] # == y = data[:, -1]
    result_index = None
    
    # this is all possible features in list
    feature_left_list = list(range(width-1))
    selected_index = []
    
    for i in range(max_var):
        rss_list = []
        # Choice an best index in each iteration
        index_chosen = None
        arg_min = None
        for f in feature_left_list:
            # data set up:
            data_index_list = selected_index + [f] + [width-1]
            data_i = np.take(data, data_index_list, axis = 1)
            # send to train weight:
            weights = gradient_descent(data_i, alpha, iterations)
            # last colum is bias, need to presever that in x matrix:
            data_x = np.copy(data_i)
            data_x[:, -1] = 1
            predicted_y = np.matmul(data_x, weights)
            rss_i = np.sum((y - predicted_y)**2)
            rss_list.append(rss_i)
            arg_min = np.argmin(rss_list)
        index_chosen = feature_left_list.pop(arg_min)
        selected_index.append(index_chosen)
    result_index = np.array(selected_index)
    return result_index
"""
def fs(data, max_var):
    ####
    print("================================== called forward_selection(", data.shape, "  , ", max_var," )")
    ####
    alpha = 0.1
    iterations = 200
    result_index = np.zeros(max_var)
    ####
    y = data[:, data.shape[1]-1]
    feature_left_list = list(range(data.shape[1]-1))
    # feature_pool = data[:, feature_left_list]
    selected = list([])
    for i in range(max_var):
        modi_data = np.zeros((data.shape[0], i+2))
        modi_data[:, i+2-1] = y
        print("modi_data[1]",modi_data[0:2])
        if (selected!=None):
            tem = [data[: , s] for s in selected]
            modi_data[:, i] = tem
        rss_array = list([])
        for index in feature_left_list:
            modi_data[:, i+2-2] = data[:,index]
            weight = gradient_descent(modi_data, alpha, iterations)
            modi_data[:, -1] = 1
            predicted_y = np.matmul(modi_data, weight)
            rss_array.append(sum((predicted_y - y)**2))
        min_rss = min(rss_array)
        arg_min = np.argmin(rss_array)
        ind_selected = feature_left_list[arg_min]
        feature_left_list.remove(ind_selected)
        result_index[i] = ind_selected
        selected.append(ind_selected)
    print(result_index)
    return result_index

"""
# =======================================
# forward_result = fs(golf_data_standardized, 5)  # Implement me!
forward_result = forward_selection(golf_data_standardized, 5)

# Step 5: Implement Backward Elimination
def backward_elimination(data, max_var):
    """
    Computes the top max_var number of features by backward elimination

    :param numpy.ndarray data: numpy.ndarray data: A (n, m) numpy array, where n = number of examples and
    m=(features + 1). The last column is the label y for that example
    :type max_var: integer
    :returns A (max_var,) numpy array whose values are the features that were selected
    :rtype: numpy.ndarray
    """
        # default varlue:
    alpha = 0.1
    iterations = 200
    # Useful latter:
    lenth = data.shape[0]
    width = data.shape[1]
    # y is the last colum in the data
    y = data[:, width-1] # == y = data[:, -1]
    result_index = None
    
    # this is all possible features in list
    feature_left_list = list(range(width-1))
    copy_feature_list = []
    selected_index = []
    
    for i in range(width-1-max_var):
        rss_list = []
        # Choice an best index in each iteration
        index_chosen = None
        arg_max = None
        for k in range(len(feature_left_list)):
            # data set up:
            copy_feature_list = list(np.copy(feature_left_list))
            copy_feature_list.pop(k)
            data_index_list = copy_feature_list + [width-1]
            data_i = np.take(data, data_index_list, axis = 1)
            # send to train weight:
            weights = gradient_descent(data_i, alpha, iterations)
            # last colum is bias, need to presever that in x matrix:
            data_x = np.copy(data_i)
            data_x[:, -1] = 1
            predicted_y = np.matmul(data_x, weights)
            rss_i = np.sum((y - predicted_y)**2)
            rss_list.append(rss_i)
            arg_min = np.argmin(rss_list)
        index_chosen = feature_left_list.pop(arg_min)
        # selected_index.append(index_chosen)
    result_index = np.array(feature_left_list)
    return result_index


backward_result = backward_elimination(golf_data_standardized, 5)  # Implement me!


# Step 6: Implemnt Gradient Descent with Lasso
def gradient_descent_lasso(data, alpha, iterations, penalty):
    """
    Performs gradient descent using the supplied data, learning rate, number of iterations, and LASSO penalty (lambda).
    The code for this should be structurally the same as gradient_descent, with the exception that after each iteration
    you will pass the weight vector through the LASSO projection. Start with weights = the zero vector.

    :param numpy.ndarray data: A (n, m) numpy array, where n = number of examples and m=(features + 1). The last column
    is the label y for that example
    :param float alpha: A real value corresponding to the step size for each descent step
    :param int iterations: The total number of iterations (steps) to take
    :param float penalty: A real positive value representing the LASSO penalty (lambda)
    :returns A (m,) numpy array of the final weight vector after the LASSO gradient descent process
    :rtype: numpy.ndarray
    """
    # print("data.shape", data.shape)
    # last one do not touch, it is the bias:
    w_final = np.zeros(data.shape[1])
    for i in range(iterations):
        w_final = w_final - alpha*gradient(data, w_final)
        for k in range(len(w_final)-1):
            if w_final[k] > penalty:
                w_final[k] = w_final[k] - penalty
            elif w_final[k] < (-penalty):
                w_final[k] = w_final[k] + penalty
            else:
                w_final[k] = 0
    return w_final
    #
    #
