"""
Description

"""
import matplotlib.pyplot as plt
from mysklearn.mypytable import MyPyTable
import numpy as np

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

def generate_freq_chart(table, col_name, xl, yl, min_value=0):
    """Function to make a frequency chart based on column in a table
    
    """
    temp_dict = get_col_items_count(table, col_name)
    temp_dict = {k:v for k, v in temp_dict.items() if v > min_value}
    make_freq_chart_from_dict(temp_dict, xl, yl)

def generate_scatter_plot(table, col_name, xl, yl, title, min_value=0):
    """Function to create a scatter plot based on column name
    
    """
    temp_dict = get_col_items_count(table, col_name)
    temp_dict = {k:v for k, v in temp_dict.items() if v > min_value}
    plt.scatter(temp_dict.keys(), temp_dict.values())
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.title(title)
    plt.show()

def generate_pie_chart_from_column(table, col_name, data_labels, title):
    l = table.get_column(col_name)
    freq_dict = {item:0 for item in l}

    for item in l:
        freq_dict[item] += 1

    freq_dict = dict(sorted(freq_dict.items(), key=lambda x: x[0], reverse=False))
    vals = list(freq_dict.values())
    
    s = sum(vals)

    labels = ["%s - %1.1f%%"%(data_labels[i], (vals[i] / s *100)) for i in range(len(data_labels)) if vals[i] > 0]
    
    plt.pie(vals, labels=labels)
    plt.title(title)
    plt.show()

def make_success_and_failure_chart(success_table, failure_table, col_name, min_val, fig_num):
    success_dict = get_col_items_count(success_table, col_name)
    success_dict = {k:v for k, v in success_dict.items() if v > min_val} 
    
    failure_dict = get_col_items_count(failure_table, col_name)
    failure_dict = {k:v for k, v in failure_dict.items() if k in success_dict.keys()}
    success_dict = {k:v for k, v in success_dict.items() if k in failure_dict.keys()}
    
    success_dict = dict(sorted(success_dict.items(), key=lambda x: x[1], reverse=True))
    failure_dict = dict(sorted(failure_dict.items(), key=lambda x: x[1], reverse=True))
    
    
    plt.figure(figsize=(8, 5))
    X_axis = np.arange(len(success_dict.keys()))
    
    plt.bar(X_axis - 0.2, list(success_dict.values()), 0.4, label='Success')
    plt.bar(X_axis + 0.2, list(failure_dict.values()), 0.4, label='Failures')
    
    plt.xticks(range(len(success_dict)), success_dict.keys(), rotation=90)
    
    plt.xlabel("%s instances with more than %i attempts"%(col_name, min_val))
    plt.ylabel("Number of Climbs")
    plt.title("Fig %i: Summit Success and Failure Count by %s"%(fig_num, col_name))
    plt.legend()
    plt.show()