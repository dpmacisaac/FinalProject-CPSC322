'''
Author: Dominic MacIsaac
functions:
    binary_precision_score
    binary_recall_score
    binary_f1_score

    train_test_split
    kfold_split
    stratified_kfold_split
    bootstrap_sample
    accuracy_score
    confusion_matrix
'''
import numpy as np # use numpy's random number generation

from mysklearn import myutils

def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        precision(float): Precision of the positive class

    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    if labels is None:
        labels = np.unique(y_true)

    if pos_label is None:
        pos_label = labels[0]

    tp_count = 0
    fp_count = 0

    for i in range(len(y_true)):
        if y_true[i] == y_pred[i] and y_pred[i] == pos_label:
            tp_count += 1
        elif y_pred[i] == pos_label:
            fp_count += 1

    try:
        precision = tp_count / (tp_count + fp_count)
    except:
        precision = 0
    
    return precision

def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        recall(float): Recall of the positive class

    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    if labels is None:
        labels = np.unique(y_true)

    if pos_label is None:
        pos_label = labels[0]

    tp_count = 0
    fn_count = 0

    for i in range(len(y_true)):
        if y_true[i] == y_pred[i] and y_pred[i] == pos_label:
            tp_count += 1
        elif y_pred[i] != pos_label and y_true[i] == pos_label:
            fn_count += 1

    try:
        recall = tp_count / (tp_count + fn_count)
    except:
        recall = 0

    return recall
def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        f1(float): F1 score of the positive class

    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    precision = binary_precision_score(y_true, y_pred, labels, pos_label)
    recall = binary_recall_score(y_true, y_pred, labels, pos_label)

    try:
        f1 = 2 * (precision * recall) / (precision + recall)
    except:
        f1 = 0

    return f1 

#####################################################################
# PA5
#####################################################################

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)

    Note:
        Loosely based on sklearn's train_test_split():

    """
    np.random.seed(random_state)
    indexes_for_testing = []
    if test_size < 1:
        test_size_amt = int(np.ceil(len(X) * test_size))
    else:
        if test_size > len(X):
            test_size_amt = len(X)
        else:
            test_size_amt = int(test_size)
    if shuffle is False:
        for i in range(test_size_amt):
            indexes_for_testing.append(len(X)-1-i)
    else:
        for i in range(test_size_amt):
            added_index = False
            while added_index is False:
                num = np.random.randint(0,len(X))
                if indexes_for_testing.__contains__(num) is False:
                    added_index = True
                    indexes_for_testing.append(num)
    X_train, X_test = myutils.split_data(X,indexes_for_testing)
    y_train, y_test = myutils.split_data(y,indexes_for_testing)
    return X_train, X_test, y_train, y_test

def kfold_split(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    folds = []
    for _ in range(n_splits):
        folds.append(([],[]))
    if shuffle is False:
        for i in range(len(X)):
            place = i%n_splits
            folds[place][1].append(i)
    else:
        if random_state is not None:
            random_state *= 5
        np.random.seed(random_state)
        added = []
        current = 0
        for _ in range(len(X)):
            place = current%n_splits
            added_yet = False
            while added_yet is False:
                num = np.random.randint(0,len(X))
                if added.__contains__(num) is False:
                    folds[place][1].append(num)
                    added.append(num)
                    added_yet = True
            current +=1
    folds = myutils.fill_in(folds, len(X))
    return folds

def stratified_kfold_split(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    grouped_vals, _ = myutils.groupby_vals(X,y)
    folds = []
    for _ in range(n_splits):
        folds.append(([],[]))
    if shuffle is False:
        current = 0
        for i in range(len(grouped_vals)):
            for j in range(len(grouped_vals[i])):
                place = current%n_splits
                folds[place][1].append(grouped_vals[i][j])
                current +=1
    else:
        if random_state is not None:
            random_state *= 4
        np.random.seed(random_state)
        added = []
        current = 0
        for i in range(len(grouped_vals)):
            for _ in range(len(grouped_vals[i])):
                place = current%n_splits
                added_yet = False
                while added_yet is False:
                    num = grouped_vals[i][np.random.randint(0,len(grouped_vals[i]))]
                    if added.__contains__(num) is False:
                        folds[place][1].append(num)
                        added.append(num)
                        added_yet = True
                current +=1
    folds = myutils.fill_in(folds, len(X))
    return folds

def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results

    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
        Sample indexes of X with replacement, then build X_sample and X_out_of_bag
            as lists of instances using sampled indexes (use same indexes to build
            y_sample and y_out_of_bag)
    """
    X_sample = []
    X_out_of_bag = []
    y_sample = []
    y_out_of_bag = []

    np.random.seed(random_state)
    if n_samples is not None:
        sample_size = n_samples
    else:
        sample_size = len(X)
    indexes_grabbed = []
    for _ in range(sample_size):
        indexes_grabbed.append(np.random.randint(0,len(X)))
    for i in range(len(X)):
        if indexes_grabbed.__contains__(i) is False:
            X_out_of_bag.append(X[i])
            if y is not None:
                y_out_of_bag.append(y[i])
        else:
            X_sample.append(X[i])
            if y is not None:
                y_sample.append(y[i])

    if y is not None:
        return X_sample, X_out_of_bag, y_sample,y_out_of_bag

    return X_sample, X_out_of_bag, None,None

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    matrix = []
    for i in range(len(labels)):
        matrix.append([])
        for _ in range(len(labels)):
            matrix[i].append(0)
    for i in range(len(y_true)):
        if y_pred[i] is not None:
            matrix[labels.index(y_true[i])][labels.index(y_pred[i])] += 1
    return matrix

def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).

    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    correct = [y_true[i] for i in range(len(y_pred)) if y_true[i] == y_pred[i]]

    if correct and normalize:
            return float(len(correct) / len(y_pred))
    
    return len(correct)

