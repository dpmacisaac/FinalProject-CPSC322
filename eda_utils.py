"""
Description

"""
import matplotlib.pyplot as plt
from mysklearn.mypytable import MyPyTable


def get_col_items_count(table, col_name):
    """Utility function to get the count of each item in a 
        column from a table as a dict
    
    args:
        table: MyPyTable to use
        col_name: column to get frequencies from
    returns:
        dictionary with (item, count)
    """
    # Getting platform frequencies:
    col = table.get_column(col_name)
    freq_dict = {item:0 for item in col}

    for item in col:
        freq_dict[item] += 1

    return dict(sorted(freq_dict.items(), key=lambda x: x[1], reverse=True))

def make_freq_chart_from_dict(dict, xl, yl):
    """Creates a bar chart from a dictionary of (item, count)
    args: 
        dict: dict to make chart form
        xl: x-axis label
        yl: y-axis label
    
    """
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(dict)), dict.values())
    plt.xticks(range(len(dict)), dict.keys(), rotation=90)
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.show()
