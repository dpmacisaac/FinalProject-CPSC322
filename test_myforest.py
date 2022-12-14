'''
Testing for MyRandomForestClassifier

    Used the interview dataset to test
    the trees that were used and picked in the RandomForestClassifier.

    Used made up instances and the handwritten results to test
    the predict method

    Also tested both M and F in fit()

'''

from mysklearn.myclassifiers import MyRandomForrestClassifier
import numpy as np

X = [
    ["Senior", "Java", "no", "no"],
    ["Senior", "Java", "no", "yes"],
    ["Mid", "Python", "no", "no"],
    ["Junior", "Python", "no", "no"],
    ["Junior", "R", "yes", "no"],
    ["Junior", "R", "yes", "yes"],
    ["Mid", "R", "yes", "yes"],
    ["Senior", "Python", "no", "no"],
    ["Senior", "R", "yes", "no"],
    ["Junior", "Python", "yes", "no"],
    ["Senior", "Python", "yes", "yes"],
    ["Mid", "Python", "no", "yes"],
    ["Mid", "Java", "yes", "no"],
    ["Junior", "Python", "no", "yes"]
]

y = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]

X_test  = [
           ["Senior", "Java", "no", "yes"], #False
           ["Junior", "Python", "yes", "yes"], #True
           ["Senior", "Java", "yes", "no"], #False
           ["Mid", "R", "no", "no"] # True
          ]

y_sol = ["False","True","False", "True"]

trees_sols =\
    [
        ['Attribute', 'att0', 
            ['Value', 'Junior', 
                ['Attribute', 'att1', 
                    ['Value', 'yes', 
                        ['Leaf', 'True', 2, 3]
                    ], 
                    ['Value', 'no', 
                        ['Leaf', 'False', 1, 2]
                    ]
                ]
            ], 
            ['Value', 'Senior', 
                ['Leaf', 'False', 1, 8]
            ], 
            ['Value', 'Mid', 
                ['Leaf', 'True', 2, 8]
            ]
        ],
        ['Attribute', 'att1', 
            ['Value', 'Java', 
                ['Leaf', 'False', 2, 8]
            ], 
            ['Value', 'Python', 
                ['Leaf', 'True', 1, 2]
            ], 
            ['Value', 'R', 
                ['Attribute', 'att0', 
                    ['Value', 'Senior', 
                        ['Leaf', 'True', 1, 4]
                    ], 
                    ['Value', 'Junior', 
                        ['Leaf', 'False', 1, 2]
                    ], 
                    ['Value', 'Mid', 
                        ['Leaf', 'True', 1, 4]
                    ]
                ]
            ]
        ]
    ]

def test_myforest_fit():
    N = 20
    M = 2
    F = 2
    
    interview_forest = MyRandomForrestClassifier(N,M,F,seed=5)
    interview_forest.fit(X,y)

    assert len(interview_forest.trees) == M

    # ensures that the trees are being created with X vals that have the same amount of
    # attributes as F
    # also tests that the amount of data passed to each tree is roughly 2/3's the len
    # of the total datasest
    i = 0
    for tree in interview_forest.trees: 
        assert len(tree.X_train[0]) == F
        assert np.isclose(len(tree.X_train), len(X)*(2/3), 2)
        assert str(tree.tree) == str(trees_sols[i])
        i +=1


def test_myforest_predict():
    N = 20
    M = 2
    F = 2
    
    interview_forest = MyRandomForrestClassifier(N,M,F,seed=5)
    interview_forest.fit(X,y)

    y_pred = interview_forest.predict(X_test)
    assert y_pred == y_sol


