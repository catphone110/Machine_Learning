import numpy as np
import math

def split_by_feature(objects, feature_index):
    '''
    Groups the given objects by feature 'feature_index'

    :param list objects: The list of objects to be split
    :param int feature_index: The index of the feature by which to split
    :return: A dictionary of (feature value) => (list of objects with that feature value)
    :rtype: dict
    '''

    # find all the distinct values of this feature
    distinct_values = set([object[0][feature_index] for object in objects])

    # create a mapping of value => list of objects with that value
    mapping = {
        value: [object for object in objects if object[0][feature_index] == value]
        for value in distinct_values
    }
    return mapping


def group(objects):
    '''
    Groups the given objects by their labels, returning the weighted count of each label (i.e. it returns the sum
    of the weights of all objects with this label, not the count of the objects)

    :param list objects: The list of objects to be grouped
    :return: A dictionary of (label value) => (sum of object weights with this label)
    :rtype: dict
    '''

    return {
        label: sum([object[2] for object in objects if object[1] == label])
        for label in set([object[1] for object in objects])
    }


def split_quality(before_split, split_results, evaluation_function):
    '''
    Takes the before and after of a split, along with an evaluation function, and returns the quality of this
    split (higher values mean better splits)

    :param list before_split: The full list of objects before the split
    :param dict split_results: The dictionary of (feature_value) => (list of objects with that feature value)
    that was returned from split_by_feature
    :return: The result of the evaluation function for this split
    :rtype: float
    '''

    # group these both by the label => count
    return evaluation_function(group(before_split),
                               [group(subset) for feature_value, subset in split_results.items()])


def dominant_label(objects):
    '''
    Accepts a list of objects and returns the most common label. It takes into account the object weights. Ties
    are broken in an undefined manner.

    :param list objects: The list of objects
    :return: The label that appeared most frequently in these objects
    :rtype: object
    '''

    grouping = group(objects)
    return (list(sorted(grouping.items(), key=lambda group: -group[1])) or [None])[0][0]


def train_tree(objects, evaluation_function, max_depth=None):
    '''
    Trains a decision tree with a specific split evaluation function and a maximum depth

    :param list objects: The list of training objects
    :param callable evaluation_function: The function to be used to evaluate the quality of the splits (either
    gini_gain or information_gain)
    :param int max_depth: The maximum depth (number of tests) to be applied to the data
    :return: A tree object. In cases of a homogeneous set of data or when max_depth=0, this will return an object
    representing the label in this data set. In other cases, it will return a dictionary of {feature => (the
    feature index to split on), actions => (a dictionary of possible feature values => the subsequent tree for
    that value)}.
    :rtype: object
    '''

    # if there are no objects to split, we can't make a guess
    if len(objects) == 0:
        return None

    # if we've hit the max_depth limit, just return the majority label at this point
    if max_depth == 0:
        return dominant_label(objects)

    # find the next split by looping through all features. we will assume that 'objects' is a
    # square table, so the feature indices can be taken from the first example
    best_quality = 0
    best_feature = None
    best_split = None
    for feature_index in range(len(objects[0][0])):
        split_results = split_by_feature(objects, feature_index)

        # evaluate our dataset after the split
        quality = split_quality(objects, split_results, evaluation_function)

        if quality > best_quality + 0.00000001: # this is to eliminate ambiguities with rounding issues. if the two are close, it will always pick the smallest feature index
            best_quality = quality
            best_feature = feature_index
            best_split = split_results

    # perform whichever split we determined was best
    if best_feature is None:
        # no split made any improvement. return the dominant label in this set
        return dominant_label(objects)
    else:
        # a split was made. drill down further
        return {
            "feature": best_feature,
            "actions": {
                feature_value: train_tree(objects, evaluation_function, None if max_depth is None else max_depth - 1)
                for feature_value, objects in best_split.items()
            }
        }


def evaluate_single(tree, object):
    '''
    Obtains a prediction for a given object using the given decision tree. Note that although the object's true
    label is technically passed through 'object', it is never used.

    :param object tree: The tree being used to evaluate the object
    :param tuple object: The object to be evaluated. The ground truth label, object[1], will always be set to None.
    :return: The label corresponding to the prediction for this class, or None if the decision tree did not output a label
    :rtype: object
    '''

    if tree is None or not isinstance(tree, dict):
        # we've reached a leaf!
        return tree
    elif object[0][tree["feature"]] not in tree["actions"]:
        # we didn't see this feature value when training the tree, so just return None for now
        return None
    else:
        # recurse!
        return evaluate_single(tree["actions"][object[0][tree["feature"]]], object)


# have them implement this
def gini_impurity_calculation(dic_data):
    # IG(p) = sum_i_in_Kclass(pi-(1-pi))
    #       = 1 - sum_K(pi^2)
    count = np.array([])
    for key, value in dic_data.items():
        # print("key", key)
        # print("value", value)
        count = np.append(count, value)
        # print("count",count)
    # print(np.array(count))
    count = count/sum(count)
    # print("count", count)
    # print("1-sum((count)**2)", 1-sum((count)**2))
    return 1-sum(count**2)


def gini_gain(pre_split, post_split):
    '''
    Evaluates the quality of a split using the Gini Impurity metric

    :param dict pre_split:
                A dictionary of (label) => (weighted count)
                corresponding to the number of instances of this label before the split
    :param list post_split:
                A list of dictionaries following the same format as pre_split.
                Each entry in this list corresponds to the new distribution after the split.
    :return: A real (non-negative) number, where a higher value indicates a higher purity post-split compared to
    pre-split
    :rtype: float
    '''
    # get_all_values_from_dict = np.fromiter(iter(post_split.values()), dtype = float)
    parent_GINI_val = gini_impurity_calculation(pre_split)
    # print(parent_GINI_val)
    gini_child_sum = 0
    sum_n_post_split = np.fromiter(iter(pre_split.values()), dtype = float)
    for dic_i in post_split:
        child_i_gini_val = gini_impurity_calculation(dic_i)
        sum_ni = np.fromiter(iter(dic_i.values()), dtype = float)
        ni_n = sum(sum_ni)/sum(sum_n_post_split)
        gini_child_sum = gini_child_sum + ni_n*child_i_gini_val
        # print("key", key)
        # print("value", value)
        # count = np.append(count, value)
        # print("count",count)
    # print(np.array(count))
    # count = count/sum(count)
    # print("parent_GINI_val", parent_GINI_val)
    # print("gini_child_sum", gini_child_sum)
    # print("==========gini======", parent_GINI_val - gini_child_sum)
    return parent_GINI_val - gini_child_sum

# have them implement this

def entropy(dic_data):
    counts_list = np.fromiter(iter(dic_data.values()), dtype = float)
    pi = counts_list/sum(counts_list)
    return -(sum(pi*np.log2(pi)))


def information_gain(pre_split, post_split):
    '''
    Evaluates the quality of a split using the Information Gain metric

    :param dict pre_split: A dictionary of (label) => (weighted count) corresponding to the number of instances
    of this label before the split
    :param list post_split: A list of dictionaries following the same format as pre_split. Each entry in this list
    corresponds to the new distribution after the split.
    :return: A real (non-negative) number, where a higher value indicates a higher purity post-split compared to
    pre-split
    :rtype: float
    '''
    parent_entropy = entropy(pre_split)
    child_w_sum_entropy = 0
    n_sum = np.fromiter(iter(pre_split.values()), dtype = float)
    for dic_i in post_split:
        child_en = entropy(dic_i)
        sum_ni = np.fromiter(iter(dic_i.values()), dtype = float)
        ni_n = sum(sum_ni)/sum(n_sum)
        child_w_sum_entropy = child_w_sum_entropy + ni_n*child_en
        # count = np.append(count, value)
    # count = count/sum(count)
    return parent_entropy - child_w_sum_entropy


def evaluate_tree(tree, objects):
    '''
    Evaluates the weighted accuracy of a decision tree on a list of objects. When calling 'evaluate_single', the
    true label will not be passed.

    :param object tree: The tree being used to evaluate the objects
    :param list objects: The objects to be used as the test set
    :return: A real number between 0 and 1, where 1 corresponds to a perfect ability to predict
    :rtype: float
    '''

    errors = 0
    total = 0
    for object in objects:
        total += object[2]
        if evaluate_single(tree, (object[0], None) + object[2:]) != object[1]:
            errors += object[2]

    return 1 - errors / total


def adaboost(objects, iterations, stump_depth):
    '''
    Trains a set of decision stumps using AdaBoost

    :param list objects: The training data
    :param int iterations: How many decision stumps we should train
    :param int stump_depth: The depth of each tree trained
    :return: A list of tuples (tree, weight), which is the model learned
    :rtype: list
    '''
    
    # train tree each iteration
    # get error each time
    # error = 1 - evaluate_tree(tree_i)
    # alpha = log((1-error)/error) - log(#of class -1)
        # where alpha is the week classfiler weight

    # adaboost_tuple is what we wanted to return
    # adaboost_tuple contains a list of (tree_i, alpha)
    adaboost_tuple = []
    for i in range(0, iterations):
        # get new tree each time
        # use train_tree(objects, evaluation_function, max_depth=None)
        # using information_gain as evalutor
        tree_i = train_tree(objects, information_gain, stump_depth)
        # calculating error, where error = 1 - evaluate_tree(tree_i)
        # print(" evaluate_tree(tree_i)", evaluate_tree(tree_i))
        error = 1 - evaluate_tree(tree_i, objects)
        # alpha = log((1-error)/error) + log(#of class -1)
        alpha = np.log((1-error)/error) + np.log(4-1)
        # object = data
        
        for j in range(len(objects)):
            # evaluate_single(tree, object):
            # Obtains a prediction for a given object using the given decision tree. Note that although the object's true
            # label is technically passed through 'object', it is never used.
            #
            # this is to check label equal or not
            #print("tuple:" inside_tuple)
            
            if evaluate_single(tree_i, (objects[j][0], None) + objects[j][2:]) != objects[j][1]:
                # if label is not consistent
                # update weight: objects[obj][2] change
                # weight new = (weight old) * e^(alpha)
                objects[j] = (objects[j][0], objects[j][1], objects[j][2]*np.exp(alpha)) 
        adaboost_tuple.append((tree_i, alpha))
    return adaboost_tuple

def evaluate_adaboost_single(trees, object):
    '''
    Takes the learned AdaBoost model and an object and computes its predicted class label

    :param list trees: The AdaBoost model returned from adaboost()
    :param tuple object: The test example for which to obtain a prediction. The ground truth label, object[1],
    will always be set to None.
    :return: The predicted class label for 'object'
    :rtype: object
    '''
    # train tree each iteration
    # get error each time
    # error = 2 - evaluate_tree(tree_i)
    # alpha = log((1-error)/error) - log(#of class -1)
        # where alpha is the week classfiler weight
    error = 0
    alpha = 0
    # adaboost_object is what we wanted to return
    # adaboost_object contains (tree_i, alpha)
    each_label_obj = dict()
    # evaluate each tree_i in trees
    for(tree_i, alpha_i) in trees:
        label = evaluate_single(tree_i, object)
        # if label is in dic
        if label in each_label_obj:
            # if it is in
            # each_label_obj[label] is weight corresponding label
            each_label_obj[label] = each_label_obj[label] + alpha_i
        else:
            each_label_obj[label] = alpha_i

    # find label have max weight
        # max(dic, key = dic.get) : returns key that have max value in dic
    best_label = max(each_label_obj, key = each_label_obj.get)
    return best_label



def evaluate_adaboost(trees, objects):
    '''
    Evaluates the weighted accuracy of a AdaBoost model on a list of objects. When calling 'evaluate_adaboost_single', the
    true label will not be passed.

    :param list trees: The AdaBoost model returned from adaboost()
    :param list objects: The objects to be used as the test set
    :return: A real number between 0 and 1, where 1 corresponds to a perfect ability to predict
    :rtype: float
    '''

    errors = 0
    total = 0
    for object in objects:
        total += object[2]
        if evaluate_adaboost_single(trees, (object[0], None) + object[2:]) != object[1]:
            errors += object[2]

    return 1 - errors / total
