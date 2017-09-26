import numpy as np
def MLE(data, labels):
    """
    Please follow the format of fA_params and fA_pi, and return the
    result in the format of (fA_params,fA_pi)
    :type data: 2D numpy array
    :type labels: 1D numpy array
    :rtype: tuple
    """
    params = np.zeros((2,3,13))
    pi = np.zeros(3)
    label1_x = np.array([])
    label2_x = np.array([])
    label3_x = np.array([])
    for i in range(len(labels)):
        if labels[i] == 1:
            label1_x = np.append(label1_x, data[i])
        elif labels[i] == 2:
            label2_x = np.append(label2_x, data[i])
        else:
            label3_x = np.append(label3_x, data[i])

    label1_x = label1_x.reshape(-1,13)
    label2_x = label2_x.reshape(-1,13)
    label3_x = label3_x.reshape(-1,13)
    for i in range(13):
        params[0,0,i] = sum(label1_x[:,i])/len(label1_x)
        params[0,1,i] = sum(label2_x[:,i])/len(label2_x)
        params[0,2,i] = sum(label3_x[:,i])/len(label3_x)

    for i in range(13):
        params[1,0,i] = sum((label1_x[:,i] - params[0,0,i])**2)  / len(label1_x)
        params[1,1,i] = sum((label2_x[:,i] - params[0,1,i])**2)  / len(label2_x)
        params[1,2,i] = sum((label3_x[:,i] - params[0,2,i])**2)  / len(label3_x)
    pi[0] = len(label1_x)/len(data)
    pi[1] = len(label2_x)/len(data)
    pi[2] = len(label3_x)/len(data)
    MLE_tuple = (params,)
    MLE_tuple = MLE_tuple + (pi,)
    return MLE_tuple

MLE_result = MLE(training_data, training_label)
MAP_result = MLE_result
best_prior = MLE_result
predictions = np.zeros((48,), dtype = int)

#=================================================
def apply(data, params, pi):
    """
    :type data: 1D numpy array
    :type params: 3D numpy array for mu and sigma^2
    :type pi: 1D numpy array for pi
    :rtype: 1D numpy array, the normalized predicted distribution
    """
    apply_pred = np.zeros((5,3))
    for i in range(5):
        x = data[i]
        # params [0, j, :] is mul
        # params [1, j, :] is variance
        # pi [j] = p(y)
        p_xi_y1 = 1/(2 * np.pi * params[1,0,:])**0.5 * np.exp(-1/2 * ((x-params[0,0,:])/(params[1,0,:])**0.5)**2)
        p_xi_y2 = 1/(2 * np.pi * params[1,1,:])**0.5 * np.exp(-1/2 * ((x-params[0,1,:])/(params[1,1,:])**0.5)**2)
        p_xi_y3 = 1/(2 * np.pi * params[1,2,:])**0.5 * np.exp(-1/2 * ((x-params[0,2,:])/(params[1,2,:])**0.5)**2)
        y1 = 1
        y2 = 1
        y3 = 1
        for j in range(13):
            y1 = y1 * p_xi_y1[j]
            y2 = y2 * p_xi_y2[j]
            y3 = y3 * p_xi_y3[j]
        y1 = pi[0] * y1
        y2 = pi[1] * y2
        y3 = pi[2] * y3
        apply_pred[i, 0] = y1 / (y1 + y2 + y3)
        apply_pred[i, 1] = y2 / (y1 + y2 + y3)
        apply_pred[i, 2] = y3 / (y1 + y2 + y3)
    return apply_pred
predicted_distr = apply(training_data, MLE_result[0], MLE_result[1])


#=================================================
def MAP(data, labels, prior_params, pseudo_count):
    """
    :type data: 2D numpy array
    :type labels: 1D numpy array
    :type params: 3D numpy array for mu and sigma^2
    :type pseudo_count: 1D numpy array for pi, recall that this is fA_pi[1,:]
    :rtype:tuple, same format as MLE
    """
    # pseudo_count ,count of y==c 是 ac
    # label , count of y==c 是 Nc

    # var就是老var

    # mul_MAP = (var * mul_pior + N_c * var_pior * mul_mle)/(N_c * var_pior + var)
    # pi_MAP = (N_c + predu_count_c - 1)/(N + predu_count_total - K)
    # where K is number of class
    # print("fA_pi",fA_pi)
    # print("prior_params",prior_params)
    # ====== initialize ===============================================
    MAP_params = np.zeros((2,3,13))
    MAP_pi = np.zeros(3)
    label1_x = np.array([])
    label2_x = np.array([])
    label3_x = np.array([])
    for i in range(len(labels)):
        if labels[i] == 1:
            label1_x = np.append(label1_x, data[i])
        elif labels[i] == 2:
            label2_x = np.append(label2_x, data[i])
        else:
            label3_x = np.append(label3_x, data[i])

    label1_x = label1_x.reshape(-1,13)
    label2_x = label2_x.reshape(-1,13)
    label3_x = label3_x.reshape(-1,13)

    # N_c : numebr of [y==ci] found in data
    N_c = np.zeros(3)
    N_c[0] = len(label1_x)
    N_c[1] = len(label2_x)
    N_c[2] = len(label3_x)
    # NcTotal : numebr of ([y==ci] found in data) + (presudo c count)
    NcTotal = np.zeros(3)
    NcTotal[0] = N_c[0] + pseudo_count[1,0]
    NcTotal[1] = N_c[1] + pseudo_count[1,1]
    NcTotal[2] = N_c[2] + pseudo_count[1,2]
    # ====== Geting info from MLE =========================================
    MLE_temp_result = MLE(data, labels)
    MLE_parms = MLE_temp_result[0]
    var = MLE_parms[1]
    mul = MLE_parms[0]
    # ====== Calculating mul_MAP =========================================
    # mul_MAP = (var * mul_pior + N * var_pior * mul_mle)/(N * var_pior + var)
    # N = N_c + presudo_count_c
    pior_var = prior_params[1]
    pior_mul = prior_params[0]
    for i in range(13):
        MAP_params[0,0,i] = (var[0,i]*pior_mul[0,i] + N_c[0]*pior_var[0,i]*mul[0,i])/(N_c[0]*pior_var[0,i] + var[0,i])
        MAP_params[0,1,i] = (var[1,i]*pior_mul[1,i] + N_c[1]*pior_var[1,i]*mul[1,i])/(N_c[1]*pior_var[1,i] + var[1,i])
        MAP_params[0,2,i] = (var[2,i]*pior_mul[2,i] + N_c[2]*pior_var[2,i]*mul[2,i])/(N_c[2]*pior_var[2,i] + var[2,i])
    # ====== calculating MAP_pi ======================================
    # print("MAP_params", MAP_params)
    # pi_MAP = (N_c + predu_count_c - 1)/(N + predu_count_total - K)
    MAP_pi[0] = (N_c[0] + pseudo_count[1,0] - 1)/(len(data) + sum(pseudo_count[1]) - 3)
    MAP_pi[1] = (N_c[1] + pseudo_count[1,1] - 1)/(len(data) + sum(pseudo_count[1]) - 3)
    MAP_pi[2] = (N_c[2] + pseudo_count[1,2] - 1)/(len(data) + sum(pseudo_count[1]) - 3)
    # ====== submit MAP ===============================================
    # variance does not change from MLE to MAP for Gaussian distribution
    MAP_params[1] = var

    MAP_tuple = (MAP_params,)
    MAP_tuple = MAP_tuple + (MAP_pi,)
    return MAP_tuple
MAP_result = MAP(training_data, training_label, fA_params , fA_pi)



def predict_single_y(x, params, pi):
    p_xi_y1 = 1/(2 * np.pi * params[1,0,:])**0.5 * np.exp(-1/2 * ((x-params[0,0,:])/(params[1,0,:])**0.5)**2)
    p_xi_y2 = 1/(2 * np.pi * params[1,1,:])**0.5 * np.exp(-1/2 * ((x-params[0,1,:])/(params[1,1,:])**0.5)**2)
    p_xi_y3 = 1/(2 * np.pi * params[1,2,:])**0.5 * np.exp(-1/2 * ((x-params[0,2,:])/(params[1,2,:])**0.5)**2)

    y_pred = np.array([pi[0]*np.prod(p_xi_y1), pi[1]*np.prod(p_xi_y2), pi[2]*np.prod(p_xi_y3)])
    y_pred = y_pred/sum(y_pred)
    return np.argmax(y_pred)+1

def test_predict_single_y():
    for i in range(5):
        y = predict_single_y(training_data[i], MAP_result[0], MAP_result[1])
        print("y predicted to be ==== ", y)
        print("y in label ===== ", training_label[i])

#test_predict_single_y()
#print("apply(training_data[0:5], MAP_result[0], MAP_result[1])",apply(training_data[0:5], MAP_result[0], MAP_result[1]))

def predict_all(data_x, params, pi):
    y_pred = np.zeros(len(data_x))
    for i in range(len(data_x)):
        y_pred[i] = predict_single_y(data_x[i], params, pi)
    return y_pred

def test_predict_all_y():
    y = predict_all(training_data, MAP_result[0], MAP_result[1])
    print("y predicted to be ==== ", y)
    print("y in label ===== ", training_label)

#test_predict_all_y()
#print("apply(training_data[0:5], MAP_result[0], MAP_result[1])",apply(training_data[0:5], MAP_result[0], MAP_result[1]))

def getAccracy(y_predicted, y):
    return sum(y_predicted == y)/len(y)

def CV(training_data, training_label, prior_params, prior_pi, k):

    #[0:26]
    #[26:52]
    #[52:78]
    #[78:104]
    #[104:130]
    # NOTE: this is measure the parameters not prediction results

#======TEST kkkkk====================
    size = int(len(training_data)/k)
    culmulated_acc = 0
    for i in range(k):
        #test 2: k = 1   test[26:52]
        test_x = training_data[i*size:(i+1)*size]
        test_y = training_label[i*size:(i+1)*size]
        train_x = np.concatenate([training_data[0:i*size], training_data[(i+1)*size:len(training_data)]])
        train_y = np.concatenate([training_label[0:i*size],training_label[(i+1)*size:len(training_label)]])
        get_map_param = MAP(train_x, train_y, prior_params, prior_pi)
        y_predicted = predict_all(test_x, get_map_param[0], get_map_param[1])

        culmulated_acc = culmulated_acc + getAccracy(y_predicted, test_y)

    return culmulated_acc/k



def decided_best():
    f_acc = np.zeros(3)
    f_acc[0] = CV(training_data, training_label, fA_params, fA_pi, 5)
    fB_params = np.array([MAP_result[1],MAP_result[1]*len(training_label)])
    f_acc[1] = CV(training_data, training_label, MAP_result[0], fB_params , 5)
    f_acc[2] = CV(training_data, training_label, fC_params, fC_pi, 5)
    best_index = np.argmax(f_acc)
    print(f_acc)
    if best_index == 0:
        param_tuple = (fA_params,)
        param_tuple = param_tuple + (fA_pi[1],)
        return param_tuple
    elif best_index == 1:
        fB_pi_presudo = MLE_result[1]*len(training_label)
        param_tuple = (MLE_result[0],)
        param_tuple = param_tuple + (fB_pi_presudo,)
        return param_tuple
    else:
        param_tuple = (fC_params,)
        param_tuple = param_tuple + (fC_pi[1],)
        return param_tuple
best_prior = decided_best()


predictions = predict_all(testing_data, best_prior[0], best_prior[1])
predictions = np.array(list(map(int, predictions)))
#print(predictions)
#
##
