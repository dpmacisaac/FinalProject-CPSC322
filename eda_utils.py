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

def generate_frequency_diagram(col, x_label, ylabel, title =None):
    '''
        creates a frequency diagram based off a col
    Args:
        col (1D list of values)
        col_name(str)
        ylabel(str)
    Returns:
        shows a frequency diagram
    '''
    values, count = get_frequencies(col)
    values = convert_to_names(values)
    plt.figure()
    plt.bar(values, count)
    plt.xticks(
        rotation=90, 
        horizontalalignment='center',
        fontweight='light',
        fontsize='small')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(ylabel)
    plt.show()

def convert_to_names(col):
    '''
        converts items in col to string
    Args:
        col (1D list of items)
    Returns:
        new_list (1D list of strings) - list of items from col in string form
    '''
    new_list = []
    for item in col:
        new_list.append(str(item))
    return new_list

def get_frequencies(col):
    '''
        returns the values and frequency/count of each of those values from col
    Args:
        col(1D list of values)
    Returns:
        values(1D list of values) - list of unique values from col
        count (1D list of integers) - parallel to values, value corresponds to amt of each value in col
        - values and count are parallel arrays
    '''
    try:
        col.sort()
    except:
        pass
    values = []
    count = []

    for i in range(len(col)):
        if col[i] in values:
            count[values.index(col[i])] +=1
        else:
            values.append(col[i])
            count.append(1)

    return values,count

def get_column(table, header, col_name):
    '''
        Grabs every item in a col in a table and returns a new list with those values
    Args:
        table(2D list of values)
        header(1D list of strings) 
        col_name (string)
    Returns:
        new_table (1D list of values) : values in a column from table
    '''
    index = header.index(col_name)
    new_table = []
    for row in table:
        new_table.append(row[index])
    return new_table