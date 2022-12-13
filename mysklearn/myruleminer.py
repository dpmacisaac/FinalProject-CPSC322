from mysklearn import myutils
from copy import deepcopy
from tabulate import tabulate

class MyAssociationRuleMiner:
    """Represents an association rule miner.

    Attributes:
        minsup(float): The minimum support value to use when computing supported itemsets
        minconf(float): The minimum confidence value to use when generating rules
        X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
        rules(list of dict): The generated rules

    Notes:
        Implements the apriori algorithm
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self, minsup=0.25, minconf=0.8):
        """Initializer for MyAssociationRuleMiner.

        Args:
            minsup(float): The minimum support value to use when computing supported itemsets
                (0.25 if a value is not provided and the default minsup should be used)
            minconf(float): The minimum confidence value to use when generating rules
                (0.8 if a value is not provided and the default minconf should be used)
        """
        self.minsup = minsup
        self.minconf = minconf
        self.X_train = None
        self.rules = None

    def fit(self, X_train):
        """Fits an association rule miner to X_train using the Apriori algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)

        Notes:
            Store the list of generated association rules in the rules attribute
            If X_train represents a non-market basket analysis dataset, then:
                Attribute labels should be prepended to attribute values in X_train
                    before fit() is called (e.g. "att=val", ...).
                Make sure a rule does not include the same attribute more than once
        """
        self.X_train = deepcopy(X_train)
        self.rules = myutils.apriori(self.X_train, self.minsup, self.minconf)

    def print_association_rules(self):
        """Prints the association rules in the format "IF val AND ... THEN val AND...", one rule on each line.

        Notes:
            Each rule's output should include an identifying number, the rule, the rule's support,
            the rule's confidence, and the rule's lift
            Consider using the tabulate library to help with this: https://pypi.org/project/tabulate/
        """
        print_table = [["#", "association rule", "support", "confidence", "lift"]]



        rule_count = 1
        for rule in self.rules:
            #rule_str = "RULE #" + str(rule_count) + ": IF "
            rule_str = ""
            for i in range(len(rule["lhs"])-1):
                rule_str += str(rule["lhs"][i]) + " AND "
            rule_str += str(rule["lhs"][len(rule["lhs"])-1])
            rule_str += " THEN "

            for i in range(len(rule["rhs"])-1):
                rule_str += str(rule["rhs"][i]) + " AND "
            rule_str += str(rule["rhs"][len(rule["rhs"])-1])

            print_table.append([
                                rule_count,
                                rule_str,
                                str(round(rule["support"],4)),
                                str(round(rule["confidence"],4)),
                                str(round(rule["completeness"],4))
                                ])
            #rule_str += "\n\t"
            #rule_str += " SUPPORT=" + str(round(rule["support"],2))
            #rule_str += " CONFIDENCE=" + str(round(rule["confidence"],2))
            #rule_str += " COMPLETENESS=" + str(round(rule["completeness"],2))
            rule_count += 1
        print(tabulate(print_table))
        
        

