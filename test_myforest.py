'''
Testing for MyRandomForestClassifier

'''

from mysklearn.myclassifiers import MyRandomForrestClassifier, MyDecisionTreeClassifier

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
    N = 20
    M = 7
    F = 3
    
    interview_forest = MyRandomForrestClassifier(N,M,F)
    interview_forest.fit(X,y)
    assert len(interview_forest.trees) == M

    preds = interview_forest.predict(X_test)
    print(preds)

    assert False

def test_myforest_predict():
    assert False


['Attribute', 'att1', 
    ['Value', 'yes', 
        ['Leaf', 'False', 2, 3]], 
    ['Value', 'no', 
        ['Leaf', 'True', 1, 8]], 
    ['Value', 'R', ['Leaf', 'True', 1, 8]], 
    ['Value', 'Java', ['Leaf', 'False', 1, 8]], 
    ['Value', 'Python', ['Leaf', 'True', 1, 8]], 
    ['Value', 'Junior', ['Leaf', 'False', 1, 8]]]