'''
Testing for MyRandomForestClassifier

'''

from mysklearn.myclassifiers import MyRandomForrestClassifier

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

X_test  = [["Senior", "Java", "no", "yes"],
           ["Mid", "Java", "no", "yes"],
           ["Junior", "R", "yes", "yes"]
          ]

y_sol = ["False", "True", "True"]




def test_myforest_fit():
    N = 10
    M = 4
    F = 2
    interview_forest = MyRandomForrestClassifier(N,M,F)
    interview_forest.fit(X,y)
    y_pred = interview_forest.predict(X_test)
    assert y_pred == y_sol

def test_myforest_predict():
    assert False