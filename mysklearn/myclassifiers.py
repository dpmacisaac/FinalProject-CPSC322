'''
Author: Dominic MacIsaac
Course: CPSC 346
Simple Classifiers:
    Decision Tree
    SimpleLinearRegressionClassifier
    KNeighborsClassifier
    DummyClassifier
    Naives Bayes Classifier
'''
import copy
from mysklearn import myutils
from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """

        # here would be a good place to programmatically 
        # the header and attribute_domains
        # next I recommend stitchng X_train and y_train together
        # so the class label is at instance [-1]
        self.X_train = X_train
        self.y_train = y_train
        train = [X_train[i]+ [y_train[i]] for i in range(len(X_train))]
        self.header = []
        domain = {}
        for j in range(len(X_train[0])):
            attr_name = "att" + str(j)
            self.header.append(attr_name)
            domain[attr_name] = []
        for i in range(len(X_train)):
            for j in range(len(X_train[i])):
                curr_attr = "att" + str(j)
                if not domain[curr_attr].__contains__(X_train[i][j]):
                    domain[curr_attr].append(X_train[i][j])
        #print()
        #print("start header:", self.header)
        #print("start domain:",domain)
        #print()

        # pass by obj reference !!
        self.tree = myutils.tdidt(train, copy.deepcopy(self.header), self.header, domain, 0)
        #print("\nFinal Tree:\n", self.tree)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for instance in X_test:
            prediction = myutils.tdidt_predict(self.tree, instance,self.header)
            y_predicted.append(prediction)
        
        return y_predicted

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        rule_str = "IF "
        myutils.print_helper(self.tree, attribute_names, class_name, rule_str)

    # BONUS method
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and
        its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).

        Notes:
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        pass # TODO: (BONUS) fix this


#######################################################################
# PA2-6
#######################################################################

class MySimpleLinearRegressionClassifier:
    """Represents a simple linear regression classifier that discretizes
        predictions from a simple linear regressor (see MySimpleLinearRegressor).

    Attributes:
        discretizer(function): a function that discretizes a numeric value into
            a string label. The function's signature is func(obj) -> obj
        regressor(MySimpleLinearRegressor): the underlying regression model that
            fits a line to x and y data

    Notes:
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self, discretizer, regressor=None):
        """Initializer for MySimpleLinearClassifier.

        Args:
            discretizer(function): a function that discretizes a numeric value into
                a string label. The function's signature is func(obj) -> obj
            regressor(MySimpleLinearRegressor): the underlying regression model that
                fits a line to x and y data (None if to be created in fit())
        """
        self.discretizer = discretizer
        self.regressor = regressor

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        if self.regressor is None:
            self.regressor = MySimpleLinearRegressor()
        self.regressor.fit(X_train, y_train)

    def predict(self, X_test):
        """Makes predictions for test samples in X_test by applying discretizer
            to the numeric predictions from regressor.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted_numeric = self.regressor.predict(X_test)
        y_predicted = []
        for item in y_predicted_numeric:
            y_predicted.append(self.discretizer(item))
        return y_predicted

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """

        distances = []
        neighbor_indices = []
        for i in range(len(X_test)):
            distances_of_test = []
            for j in range(len(self.X_train)):
                try:
                    distances_of_test.append(myutils.euclidian_distance(self.X_train[j],X_test[i]))
                except TypeError:
                    if self.X_train[i] == X_test[i]:
                        distances_of_test.append(0)
                    else:
                        distances_of_test.append(1)
            test_indices = myutils.get_smallest_k(self.n_neighbors,distances_of_test)
            k_distances_of_test = []
            for index in test_indices:
                k_distances_of_test.append(distances_of_test[index])

            distances.append(k_distances_of_test)
            neighbor_indices.append(test_indices)

        return distances, neighbor_indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        _, neighbor_indices = self.kneighbors(X_test)
        y_predicted = []
        for i in range(len(neighbor_indices)):
            classes = []
            count = []
            for j in range(len(neighbor_indices[i])):
                classifier = self.y_train[neighbor_indices[i][j]]
                if classes.__contains__(classifier):
                    count[classes.index(classifier)] +=1
                else:
                    classes.append(classifier)
                    count.append(1)
            most = (-1,-1)
            for k in range(len(count)):
                if count[k] > most[0]:
                    most = (count[k],k)
            y_predicted.append(classes[most[1]])

        return y_predicted

class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.

    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()

    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """
    def __init__(self):
        """Initializer for DummyClassifier.

        """
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        classifiers = []
        count = []
        for item in y_train:
            if classifiers.__contains__(item):
                count[classifiers.index(item)] += 1
            else:
                classifiers.append(item)
                count.append(1)
        most = (-1,-1)
        for k in range(len(count)):
            if count[k] > most[0]:
                most = (count[k],k)
        self.most_common_label = classifiers[most[1]]

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for _ in X_test:
            y_predicted.append(self.most_common_label)
        return y_predicted

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.

        priors (dictionary) : dictionary matching classifiers to the amount in the dataset
        posteriors(dictionary to list of dictionaries): dictionary matching classifiers to a list
            of dictionarys, one for each column in X_train. In each of these dictionaries matches
            attribute values to the amount of those. If no value is found, it is not in that dictionary
            and predict handles it when it is not found
        """
        self.priors = None
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        self.posteriors = {}
        self.priors = [{},0]
        empty_attr_dict_list = []
        for _ in range(len(X_train[0])):
            empty_attr_dict_list.append({})

        for i in range(len(X_train)):
            if y_train[i] not in self.posteriors:
                self.posteriors[y_train[i]] = copy.deepcopy(empty_attr_dict_list)
                self.priors[0][y_train[i]] = 1
            else:
                self.priors[0][y_train[i]] +=1
            self.priors[1] += 1

            for j in range(len(X_train[0])):
                if X_train[i][j] not in self.posteriors[y_train[i]][j]:
                    self.posteriors[y_train[i]][j][X_train[i][j]] = 1
                else:
                    self.posteriors[y_train[i]][j][X_train[i][j]] += 1

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted =[]
        classifiers = self.posteriors.keys()
        classifiers_probabilities = {}
        for class_val in classifiers:
            classifiers_probabilities[class_val] = self.priors[0][class_val]/ self.priors[1]
        #print(classifiers_probabilities)
        for instance in X_test:
            result_probabilites = {}
            #print(instance)
            for class_val in classifiers:
                result = classifiers_probabilities[class_val]
                for i in range(len(instance)):
                    try:
                        #print("class_val", class_val)
                        #print(instance[i], i)
                        num = self.posteriors[class_val][i][instance[i]]
                        #print("num", num)
                        denom = self.priors[0][class_val]
                        #print("denom", denom)
                        val = num/denom
                    except KeyError:
                        val = 0
                    result *= val
                result_probabilites[class_val] = result
            #print(result_probabilites)
            y_predicted.append(myutils.find_most_naive_bays(result_probabilites, self.priors))
        #print(y_predicted)
        return y_predicted

    def print(self, mode):
        '''
            helper function to print posteriors and priors
        '''
        if mode == 0:
            print(self.posteriors)
        if mode == 1:
            print(self.priors)