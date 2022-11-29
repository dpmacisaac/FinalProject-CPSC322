'''
Author: Dominic MacIsaac
utility functions to aid myevaluations.py
'''
import copy
from math import sqrt
import math
import numpy as np # use numpy's random number generation
from tabulate import tabulate
from mysklearn import myevaluation

###############################################################################################
# PA7 Functions
###############################################################################################
def smallest_index(vals):
    '''
    args:
        vals(1D list of comparable objs)
    returns:
        index (int)
    '''
    index = 0
    smallest_val = vals[0]
    for i in range(len(vals)):
        if vals[i] < smallest_val:
            smallest_val = vals[i]
            index = i
    return index

def select_attribute(instances, attributes):
    '''
    Uses entropy to find the attribute to split on
    returns attr with smallest entropy
    '''
    partition_entropy_vals = []
    partition_entropy_classes = []
    part_count_dict = {}
    for attr in attributes:
        part_count_dict[attr] = {}
    # Adds values into dictionaries which are used to calculate the entropy
    for i in range(len(instances)):
        if not partition_entropy_classes.__contains__(instances[i][-1]):
            partition_entropy_classes.append(instances[i][-1])
            partition_entropy_vals.append(1)
        else:
            partition_entropy_vals[partition_entropy_classes.index(instances[i][-1])] += 1
        for attr in attributes:
            attr_index = int(attr[3])
            attr_val = instances[i][attr_index]
            attr_class = instances[i][-1]
            if attr_val not in part_count_dict[attr]:
                part_count_dict[attr][attr_val] = {attr_class:1}
            else:
                if attr_class not in part_count_dict[attr][attr_val]:
                    part_count_dict[attr][attr_val][attr_class] = 1
                else:
                    part_count_dict[attr][attr_val][attr_class] += 1

    # print("dict counts:")
    # print(part_count_dict)

    # print(partition_entropy_classes)
    # print(partition_entropy_vals)

    # Calculate E_Start
    e_start = 0.0
    partition_size = len(instances)
    for val in partition_entropy_vals:
        if val != 0:
            e_start = e_start - (val/partition_size)*math.log((val/partition_size),2)
    # print("e_start", e_start)

    # Calculate E_new
    e_news = []
    e_attr_vals = []
    e_find = 0.0
    for attr in attributes:
        attr_vals = part_count_dict[attr].keys()
        e_new = 0.0
        for attr_val in attr_vals:
            e_class_val = 0.0
            classes_for_attr_val = part_count_dict[attr][attr_val].keys()
            total = 0
            for class_for_attr_val in classes_for_attr_val:
                total += part_count_dict[attr][attr_val][class_for_attr_val]
            for class_for_attr_val in classes_for_attr_val:
                count = part_count_dict[attr][attr_val][class_for_attr_val]
                e_class_val = e_class_val - ((count/total)*math.log(count/total,2))
            e_new += (total/partition_size)*e_class_val
        e_news.append(e_new)
    # print(e_news)

    attribute_smallest_entropy =attributes[smallest_index(e_news)]

    # print(attribute_smallest_entropy)

    return attribute_smallest_entropy

def partition_instances(instances, attribute, header, domain):
    '''
    creates a partition based on the instances and attributes remaining.
    '''
    att_index = int(attribute[3])
    att_domain = domain[attribute]
    #print("attribute domain:", att_domain, "\n")

    partitions = {}
    for att_value in att_domain:
        partitions[att_value] = []
        for instance in instances:
            if instance[att_index] == att_value:
                partitions[att_value].append(instance)

    return partitions

def same_class_label(instances):
    '''
    returns true if all instances have the same class label
    '''
    first_label = instances[0][-1]
    for instance in instances:
        if instance[-1] != first_label:
            return False
    # get here, all the same
    return True

def tdidt(current_instances, available_attributes, header, domain, layer):
    '''
    recursive algorithm for decision tree
    '''
    # basic approach (uses recursion!!):
    #print("available attrs:", available_attributes)
    # select an attribute to split on
    split_attribute = select_attribute(current_instances,available_attributes)
    #print(layer," splitting on:", split_attribute)
    available_attributes.remove(split_attribute)
    tree = ["Attribute", split_attribute]
    # group data by attribute domains (creates pairwise disjoint partitions)

    partitions = partition_instances(current_instances, split_attribute, header, domain) # returns a dict
    #print("partitions:", partitions)

    # for each partition, repeat unless one of the following occurs (base case)

    # len of att_partition is numerator
    # len of current_instanes is denominator
    for att_value, att_partition in partitions.items():
        #print("att_val:", att_value)
        #print("att_partition:", att_partition)
        value_subtree = ["Value", att_value]
        if len(att_partition) > 0 and same_class_label(att_partition):
            #print("Case 1: all same class label")
            # CASE 1: all class labels of the partition are the same => make a leaf node
            # MAKE A LEAF NODE HERE
            value_subtree.append(["Leaf",att_partition[0][-1],len(att_partition),len(current_instances)])
        elif len(att_partition) > 0 and len(available_attributes) == 0:
            #print("Case 2: Clash :: ",att_value, " :: ", att_partition)

            # CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
            labels = []
            label_counts = []
            total = 0
            for row in att_partition:
                if row[-1] not in labels:
                    labels.append(row[-1])
                    label_counts.append(1)
                else:
                    label_counts[labels.index(row[-1])] += 1
                total += 1
            largest_cnt = 0
            largest_val = None
            for i in range(len(labels)):
                if label_counts[i] > largest_cnt:
                    largest_cnt = label_counts[i]
                    largest_val = [labels[i]]
                elif label_counts[i] == largest_cnt:
                    largest_val.append(labels[i])
            largest_val.sort()
            value_subtree.append(["Leaf",largest_val[0],largest_cnt, total])
        elif len(att_partition) == 0:
            # CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
            #print("Case 3: no instances in partition")
            # have to backtrack, overwrite the "tree" above with a leafnode
            labels = []
            label_counts = []
            total = 0
            for _, att_partition_2 in partitions.items():
                for row in att_partition_2:
                    if row[-1] not in labels:
                        labels.append(row[-1])
                        label_counts.append(1)
                    else:
                        label_counts[labels.index(row[-1])] += 1
                    total += 1
            largest_cnt = 0
            largest_val = None
            for i in range(len(labels)):
                if label_counts[i] > largest_cnt:
                    largest_cnt = label_counts[i]
                    largest_val = labels[i]

            tree = ["Leaf",largest_val,largest_cnt, total]
            #print("case3 tree:",tree, "\n\n")
            return tree
        else:
            #print("Recurrence")
            #print("tree", tree)
            subtree = tdidt(att_partition, available_attributes.copy(), header, domain, layer + 1)
            value_subtree.append(subtree)

        if tree[0] != "Leaf":
            tree.append(value_subtree)
            #print("treettt:", tree, "\n\n")

    return tree

def tdidt_predict(tree, instance, header):
    '''
    recursive algorithm for predicting based on a decision tree
    '''
    # are we at a leaf node (base case)
    # or an attribute node (need to recurse)
    info_type = tree[0] # Attribute or Leaf
    if info_type == "Leaf":
        # base case
        return tree[1] # label

    # if we are here, then we are at an Attribute node
    # we need to figure where in instance, this attribute's value is
    att_index = header.index(tree[1])
    # now loop through all of the value lists, looking for a
    # match to instance[att_index]
    for i in range(2, len(tree)):
        value_list = tree[i]
        if value_list[1] == instance[att_index]:
            # we have a match, recurse on this value's subtree
            return tdidt_predict(value_list[2], instance, header)

def print_helper(tree, attr_names, class_name, rule_str):
    '''
    recursive helper to print a decision tree rules
    '''
    if tree[0] == "Leaf":
        rule_str = rule_str[0:(len(rule_str)-5)]
        print(rule_str + " THEN " + class_name + " = "+ tree[1])
    else:
        if attr_names is not None:
            attr_index = int(tree[1][3])
            rule_str += attr_names[attr_index] + " is "
        else:
            rule_str += tree[1] + " is "
        for i in range(2, len(tree)):
            print_helper(tree[i][2], attr_names, class_name, rule_str + str(tree[i][1]) + " AND ")

    pass
###############################################################################################
# PA6 Functions
###############################################################################################
def find_most_naive_bays(results, priors):
    '''
        finds the best class from the probabilites in naive bays
        deals with multiple items with the same probabilities and multiple items
        with the same priors amount

        Args:
            results (dictionary): maps classifiers to their probabilites
            priors(array with a dictionary and a val): dictionary maps classifiers
                to the amount of them
                val is simply the total amount of values
        returns:
            val(object) classifier

    '''
    largest = []
    result_classes = results.keys()
    for result_class in result_classes:
        if largest == []:
            largest.append(result_class)
        elif results[largest[0]] < results[result_class]:
            largest = [result_class]
        elif results[largest[0]] == results[result_class]:
            largest.append(result_class)
    if len(largest) == 1:
        return largest[0]

    most_priors = []
    for class_val in largest:
        if most_priors == []:
            most_priors.append(class_val)
        elif priors[0][most_priors[0]] < priors[0][class_val]:
            most_priors = [class_val]
        elif priors[0][most_priors[0]] == priors[0][class_val]:
            most_priors.append(class_val)
    if len(most_priors) == 1:
        return most_priors[0]

    lowest = None
    for val in most_priors:
        if lowest is None:
            lowest = val
        elif val < lowest:
            lowest = val
    return lowest

def k_fold_validation_predictions(table, y_label,labels, classifier, n_size):
    '''
        returns the predicted and solutions of all folds in a kfold validation
    '''
    y = table.get_column(y_label)
    label_indexes = []
    for label in labels:
        label_indexes.append(table.column_names.index(label))
    X_py_table_simplified = table.keep_only_these_cols(label_indexes)
    folds = myevaluation.stratified_kfold_split(X_py_table_simplified.data, y, n_splits=n_size)
    all_predicted = []
    all_solution = []
    for set_of_vals in folds:
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        for index in set_of_vals[0]:
            X_train.append(X_py_table_simplified.data[index])
            y_train.append(y[index])
        for index in set_of_vals[1]:
            X_test.append(X_py_table_simplified.data[index])
            y_test.append(y[index])
        classifier_obj = classifier()
        classifier_obj.fit(X_train, y_train)
        y_predicted = classifier_obj.predict(X_test)
        all_predicted.extend(y_predicted)
        all_solution.extend(y_test)

    return all_predicted, all_solution, classifier_obj


def print_simple_confusion_matrix(matrix):
    '''
        prints a simple 2x2 confusion matrix
    '''
    print("Confusion Matrix:")
    matrix_filled = [["","","predicted","",""],
                     ["","", "pos", "neg", "total"],
                     ["actual", "pos",matrix[0][0],matrix[0][1], matrix[0][0]+matrix[0][1]],
                     ["", "neg",matrix[1][0],matrix[1][1], matrix[1][0]+matrix[1][1]],
                     ["", "total", matrix[0][0]+matrix[1][0],matrix[0][1]+matrix[1][1], matrix[0][0]+matrix[1][0]+matrix[0][1]+matrix[1][1],]]
    print(tabulate(matrix_filled))

###############################################################################################
# PA5 Functions
###############################################################################################

def split_data(data, indexes):
    '''
        takes a set of data and indexes and returns two lists: one
        with the indexes returned the other with those indexes removed

        Args:
        data (n-D list of objs)
        indexes (1D list of integers)
        Returns:
        data_1 (n-D list of objs)
        data_2 (n-D list of objs)
    '''
    data_1 = copy.deepcopy(data)
    data_2 = []
    indexes.sort()
    for index in indexes:
        data_2.append(data_1[index])
    indexes.sort(reverse=True)
    for index in indexes:
        data_1.pop(index)
    return data_1,data_2

def fill_in(folds, size_list):
    '''
    takes a list of 2 item tuples (folds) where the second items are filled with indexes
    and returns that same list with the 1st item filled with all other indexes

        Args:
            folds(list of 2 item tuples, each item in the tuple is a list)
        Returns:
            folds(list of 2 item tuples, each item in the tuple is a list)
    '''
    for i in range(len(folds)):
        for j in range(size_list):
            if folds[i][1].__contains__(j) is False:
                folds[i][0].append(j)

    return folds

def groupby_vals(X,y):
    '''
        takes list of values and a parallel list y classifiers and returns
        a 2D list of list of indexes where each sub list is indexes of a different classifier

        Args:
            X (2D list of list of objs)
            y (1D list of objs)

        Returns:
            grouped_vals(2D list of list of indexes)
            classifiers (1D list of objects)
    '''
    grouped_vals = []
    classifiers = []
    for classifier in y:
        if classifiers.__contains__(classifier) is False:
            classifiers.append(classifier)
    for _ in range(len(classifiers)):
        grouped_vals.append([])
    for i in range(len(X)):
        grouped_vals[classifiers.index(y[i])].append(i)
    return grouped_vals, classifiers

def convert_to_1d(data):
    '''
        converts 2D list into 1D list
        iterates through each item in the 2D list and appends
        it to the end of the new list
        Args:
            data (2D list of list of objects)
        Returns
            data_1D (1D list of objects)
    '''
    data_1D = []
    for row in data:
        for item in row:
            data_1D.append(item)
    return data_1D

def matrix_add(data_1, data_2):
    '''
        adds two parallel matrixes together
        Args
            data_1(2D list of list of numerics)
            data_2(2D list of list of numerics)
        Returns
            data_1(2D list of list of numerics)
    '''
    for i in range(len(data_1)):
        for j in range(len(data_1[0])):
            data_1[i][j] += data_2[i][j]
    return data_1

########################################################
#  Testing
def test_split_data():
    '''
        testing the split data function above
    '''
    X = [[0],[1],[2],[3],[4],[5]]
    indexes = [4,1,3]
    X_1,X_2 = split_data(X, indexes)
    X_1_sol = [[0],[2],[5]]
    X_2_sol = [[1],[3],[4]]
    assert np.array_equal(X_1, X_1_sol)
    assert np.array_equal(X_2, X_2_sol)

def test_fill_in():
    '''
     testing the fill in function
    '''
    folds = [([],[0,3]),([],[1,4]), ([],[2,5])]
    size_list = 6
    folds_sol = str([([1,2,4,5],[0,3]),([0,2,3,5],[1,4]), ([0,1,3,4],[2,5])])
    folds = str(fill_in(folds,size_list))
    assert folds == folds_sol

def test_groupby_vals():
    '''
        testing the group by vals function
    '''
    X = [[0],[1],[2],[3]]
    y = [1,2,2,1]
    classifiers_sol = str([1,2])
    grouped_sol = str([[0,3],[1,2]])
    grouped, classifiers = groupby_vals(X,y)
    assert str(grouped) == grouped_sol
    assert str(classifiers) == classifiers_sol

def test_convert_to_1d():
    '''
        testing convert to 1D
    '''
    data = [[1,2],[3,4,5],[6,7,8,9]]
    data_1D = str(convert_to_1d(data))
    data_1D_sol = str([1,2,3,4,5,6,7,8,9])
    assert data_1D == data_1D_sol


###############################################################################################
# PA4 Functions
###############################################################################################
def normalize(X_train, X_test = None):
    '''
        normalizes data set and testing set
        Args:
            X_train (2D list of list of numerics)
            X_test (2D list of list of numerics)
        Returns:
            X_train_normal (2D list of list of numerics) all vals between 0 and 1
            X_test_normal (2D list of list of numerics) all vals between 0 and 1
    '''
    X_train_normal = []
    for _ in X_train:
        X_train_normal.append([])
    if X_test is not None:
        X_test_normal = []
        for _ in X_test:
            X_test_normal.append([])

    for i in range(len(X_train[0])):
        col = get_col(X_train, i)
        min_x = min(col)
        max_x = max(col)
        range_x = max_x - min_x
        for j in range(len(col)):
            X_train_normal[j].append((col[j]-min_x)/range_x)
        if X_test is not None:
            col_test = get_col(X_test, i)
            for j in range(len(col_test)):
                new_x = (col_test[j]-min_x)/range_x
                if new_x > 1:
                    X_test_normal[j].append(1)
                elif new_x < 0:
                    X_test_normal[j].append(0)
                else:
                    X_test_normal[j].append(new_x)

    return X_train_normal, X_test_normal

def get_col(data, index):
    '''
    returns all items in a a col in a list

        Args
            data (2D list of list of obj)
            index (int)

        Returns
            col (1D list of obj)
    '''
    col = []
    for item in data:
        col.append(item[index])
    return col

def euclidian_distance(a_vals, b_vals):
    '''
    returns the euclidan distance of two list of vals

        Args:
            a_vals (list of numerics)
            b_vals (list of numerics)
        Returns:
            dist (float)
    '''
    count = 0.0
    for i in range(len(a_vals)):
        count+= (a_vals[i]-b_vals[i])*(a_vals[i]-b_vals[i])
    dist = sqrt(count)
    return dist

def get_smallest_k(k, vals):
    '''
        gets the k smallest vals and returns a list of indexes

        Args:
            k (int)
            vals (1D list of vals)

        Returns:
            indexes(1D list of ints)
    '''
    indexes = []

    for _ in range(k):
        smallest = (max(vals),-1)
        for j in range(len(vals)):
            if vals[j] < smallest[0]:
                already_in = False
                for index in indexes:
                    if j == index:
                        already_in = True
                if not already_in:
                    smallest = (vals[j],j)
        indexes.append(smallest[1])

    return indexes

def range_to_val(range_vals, in_val):
    '''
        takes a list of tuples and outputs the third value in the tuple if the in_val
        is inbetween the first two vals in the tuple.
    Args:
        range_vals(List of tuples)
        in_val (numeric)
    Returns:
        int
    '''
    for val in range_vals:
        if val[0] is None:
            if in_val <= val[1]:
                return val[2]
        elif val[1] is None:
            if in_val >= val[0]:
                return val[2]
        elif in_val >= val[0] and in_val < val[1]:
            return val[2]
    return -1

def remove_cols_from_table(data, cols):
    '''
        takes a 2D table and return the table with only the columns from cols
    Args:
        data (2D list of list of values)
        cols (1D list of indexes)
    Returns:
        new_data(2D list of lists of values)
    '''
    new_data = []
    for row in data:
        new_row = []
        for col in cols:
            new_row.append(row[col])
        new_data.append(new_row)
    return new_data

def doe_rating_function(val):
    '''
        calls range_to_val with doe ratings
    '''
    return range_to_val([(None,13,1),(13,15,2),(15,17,3),(17,20,4),(20,24,5),(24,27,6),\
               (27,31,7),(31,37,8),(37,44,9),(45,None,10)], val)
