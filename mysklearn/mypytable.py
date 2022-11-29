"""
Author: Dominic MacIsaac
Course: Data Science Algorithms CPSC 322
Project: 2
"""

import copy
import csv
import os
from tabulate import tabulate
from mysklearn import myutils

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests
class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):  # done
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):  # done
        #     """Prints the table in a nicely formatted grid structure.
        #     """
        print("-------------------------------------------------------------")
        print(tabulate(self.data, headers=self.column_names))
        print("-------------------------------------------------------------")
        print()

    def get_shape(self):  # done
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        if len(self.data) < 1:  # returns 0,0 if data is empty
            return 0, 0
        return len(self.data), len(self.data[0])
        # returns len of rows and then len of
        # first col which should be the same
        # for all cols

    def get_column(self, col_identifier, include_missing_values=True):  # done
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        col_out = []
        try:
            index = self.column_names.index(col_identifier)  # trys to get the
            # index of the identifier
        except ValueError as e:  # if index get fails
            print(e)
        for row in self.data:  # goes through each row
            col_out.append(
                row[index]
            )  # in each row, grabs col and adds it into col_out
        return col_out

    def convert_to_numeric(self):  # done
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for row in range(len(self.data)):
            for val in range(
                len(self.data[row])
            ):  # iterates through every val in table
                try:
                    self.data[row][val] = float(
                        self.data[row][val]
                    )  # tries to convert val
                except:  # if it fails, nothing happens, moves to next val
                    pass

    def drop_rows(self, row_indexes_to_drop):  # done
        """Remove rows from the table data.

        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        row_indexes_to_drop.sort()  # sorts indexes so drop is consistent
        indexdrop = 0  # track how many indexes have been dropped
        for index in row_indexes_to_drop:
            try:
                self.data.pop(
                    index - indexdrop
                )  # indexes that are popped are lowered if
                # values have been popped before them,
                # lowering the amount of vals in the list
                # on every iteration
                indexdrop += 1  # iterates drop
            except ValueError as e:  # if index is not correct, throws error
                print(e)

    def load_from_file(self, filename):  # done
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        openfile = open(filename, "r")
        self.data = []
        self.column_names = []  # ensures these tables are empty
        reader = csv.reader(openfile)
        self.column_names = next(reader)  # grab first line for column names
        for row in reader:  # goes through every row in reader
            self.data.append(row)  # adds it to data
        openfile.close()
        self.convert_to_numeric()
        return self

    def save_to_file(self, filename):  # done
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        outfile = open(filename, "w")
        writer = csv.writer(outfile)
        writer.writerow(self.column_names)
        writer.writerows(self.data)
        outfile.close()

    def find_duplicates(self, key_column_names):  # done
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        searchindexes = []
        for col in key_column_names:  # convert list of key col names to list of indexes
            searchindexes.append(self.column_names.index(col))
        duplicates = []
        existing = []
        for x in range(len(self.data)):
            item = ""  # create new item for comparison
            for y in searchindexes:
                item += str(self.data[x][y])  # creates a string that represents
                # the values at each key index in a row
            if existing.__contains__(item):  # if the item is already in existing,
                # then it is a duplicate
                duplicates.append(x)
            else:  # if not a duplicate, it is added to the existing list
                existing.append(item)
        return duplicates

    def remove_rows_with_missing_values(self, empty_val="NA"):  # done
        """Remove rows from the table data that contain a missing value ("NA").

            args:
                empty_val: string that marks there is a missing val in the dataset
                    some datasets have different empty strings
        """
        new_list = []
        for row in range(len(self.data)):
            add_val = True
            for val in range(len(self.data[row])):
                if self.data[row][val] == empty_val:
                    add_val = False
            if add_val:
                new_list.append(self.data[row])
        self.data = new_list
        """
        removelist = []
        for row in range(len(self.data)):
            for val in range(len(self.data[row])):
                if self.data[row][val] == empty_val:
                    removelist.append(row)
        self.drop_rows(removelist)
        """

    def replace_missing_values_with_column_average(self, col_name):  # done
        """For columns with continuous data, fill missing values in a column
            by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        count = 0
        totalval = 0
        indexy = self.column_names.index(col_name)
        col_to_check = self.get_column(col_name)
        for val in col_to_check:
            try:
                if val != "NA":
                    count += 1
                    totalval += val
            except:
                for x in range(len(self.data[0])):
                    if self.data[x][indexy] == "NA":
                        self.data[x][indexy] = val
                return
        avg = totalval / count
        for x in range(len(self.data[0])):
            if self.data[x][indexy] == "NA":
                self.data[x][indexy] = avg

    def compute_summary_statistics(self, col_names):  # done
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
            min: minimum of the column
            max: maximum of the column
            mid: mid-value (AKA mid-range) of the column
            avg: mean of the column
            median: median of the column

        Args:
            col_names(list of str): names of the numeric columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]

        Notes:
            Missing values should in the columns to compute summary stats
                for should be ignored.
            Assumes col_names only contains the names of columns with numeric data.
        """
        statscolumn = ["attribute", "min", "max", "mid", "avg", "median"]
        statsdata = []
        statswithnomissing = MyPyTable(self.column_names, self.data)
        statswithnomissing.remove_rows_with_missing_values()
        for col in col_names:
            col_list = statswithnomissing.get_column(col)
            col_list.sort()
            try:
                min = col_list[0]
                max = col_list[len(col_list) - 1]
                mid = (max + min) / 2
                totalval = 0
                count = 0
                for val in col_list:
                    count += 1
                    totalval += val
                avg = totalval / count

                if len(col_list) % 2 == 0:
                    median = (
                        col_list[int(len(col_list) / 2)]
                        + col_list[int((len(col_list) / 2) - 1)]
                    ) / 2
                else:
                    median = col_list[int((len(col_list) / 2))]
                row = [col, min, max, mid, avg, median]
                statsdata.append(row)
            except:
                pass
        statstable = MyPyTable(statscolumn, statsdata)
        return statstable

    def perform_inner_join(self, other_table, key_column_names):  # done
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        leftIndexes = []
        rightIndexes = []
        newcol = []
        newdata = []
        for name in self.column_names:
            newcol.append(name)
        for name in other_table.column_names:
            if newcol.__contains__(name) is False:
                newcol.append(name)

        for col in key_column_names:  # convert list of key col names to list of indexes
            leftIndexes.append(self.column_names.index(col))
            rightIndexes.append(other_table.column_names.index(col))

        for leftrow in range(len(self.data)):
            leftstr = ""
            for leftcol in leftIndexes:
                leftstr += str(self.data[leftrow][leftcol])
            for rightrow in range(len(other_table.data)):
                rightstr = ""
                for rightcol in rightIndexes:
                    rightstr += str(other_table.data[rightrow][rightcol])
                if rightstr == leftstr:
                    newrow = self.deepcopymerge(
                        self.data[leftrow], other_table.data[rightrow], [], rightIndexes
                    )
                    newdata.append(newrow)
        newtable = MyPyTable(newcol, newdata)
        return newtable

    def perform_full_outer_join(self, other_table, key_column_names):  # done
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
            other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        leftIndexes = []
        rightIndexes = []
        newcol = []
        newdata = []
        for name in self.column_names:
            newcol.append(name)
        for name in other_table.column_names:
            if newcol.__contains__(name) is False:
                newcol.append(name)

        for col in key_column_names:  # convert list of key col names to list of indexes
            leftIndexes.append(self.column_names.index(col))
            rightIndexes.append(other_table.column_names.index(col))

        rightempty = ["NA" for i in range(len(other_table.data[0]) - len(rightIndexes))]
        # for x in range(len(other_table.data[0])-len(rightIndexes)):
        # rightempty.append("NA")
        leftEmpty = ["NA" for i in range(len(self.data[0]) - len(leftIndexes))]
        # for x in range(len(self.data[0])-len(leftIndexes)):
        # leftEmpty.append("NA")

        leftIndexes2, rightIndexes2 = self.indexmapping(other_table, key_column_names)

        rightPlacedBool = [False for i in range(len(other_table.data))]

        for leftrow in range(len(self.data)):
            leftstr = ""
            leftAdded = False
            for leftcol in leftIndexes:
                leftstr += str(self.data[leftrow][leftcol])
            for rightrow in range(len(other_table.data)):
                rightstr = ""
                for rightcol in rightIndexes:
                    rightstr += str(other_table.data[rightrow][rightcol])
                if rightstr == leftstr:
                    newrow = self.deepcopymerge(
                        self.data[leftrow], other_table.data[rightrow], [], rightIndexes
                    )
                    newdata.append(newrow)
                    rightPlacedBool[rightrow] = True
                    leftAdded = True
            if not leftAdded:
                newrow = self.copy_from_index(self.data[leftrow], leftIndexes2)
                newdata.append(newrow)
        for rightrow in range(len(other_table.data)):
            if not rightPlacedBool[rightrow]:
                newrow = self.copy_from_index(other_table.data[rightrow], rightIndexes2)
                newdata.append(newrow)

        newtable = MyPyTable(newcol, newdata)
        return newtable

    def deepcopymerge(self, row1, row2, skipLeft, skipRight):
        """
            merges together the left and right rows 
            when both rows have values
        """
        newrow = []
        for x1 in range(len(row1)):
            if skipLeft.__contains__(x1) is False:
                newrow.append(row1[x1])
        for x2 in range(len(row2)):
            if skipRight.__contains__(x2) is False:
                newrow.append(row2[x2])
        return newrow

    def copy_from_index(self, row, indexes):
        """
            copies the value from the row into a newrow in the order of each val in the indexes array
        """
        newrow = []
        for i in indexes:
            if i == -1:
                newrow.append("NA")
            else:
                newrow.append(row[i])
        return newrow

    def indexmapping(self, other_table, key_column_names):
        """ maps the index of each column for the left and right tables
            into an array. 
            eg for leftKeyIndex = [0,1,-1,-1]
            eg for rightKeyIndex = [-1,1,0,2]

            copy_from_index can then go through each index and add the correct value to a newrow or add NA if the index is -1 
        """
        leftIndexes = []
        rightIndex = []
        for i in range(len(self.data[0])):
            leftIndexes.append(i)
        for i in range(len(other_table.data[0]) - len(key_column_names)):
            leftIndexes.append(-1)

        leftKeyIndex = []
        rightKeyIndex = []

        for col in key_column_names:  # convert list of key col names to list of indexes
            leftKeyIndex.append(self.column_names.index(col))
            rightKeyIndex.append(other_table.column_names.index(col))

        for i in range(len(leftIndexes)):
            added = False
            for j in range(len(leftKeyIndex)):
                if leftKeyIndex[j] == i:
                    rightIndex.append(rightKeyIndex[j])
                    added = True
            if not added:
                if i < len(self.data[0]):
                    rightIndex.append(-1)
                else:
                    for k in range(len(other_table.data[0])):
                        found = False
                        for h in rightIndex:
                            if h == k:
                                found = True
                        if not found:
                            rightIndex.append(k)
        return leftIndexes, rightIndex

    def keep_only_these_cols(self,cols):
        '''
            takes a 2D table and return the table with only the columns from cols
        Args:
            data (2D list of list of values)
            cols (1D list of indexes)
        Returns:
            new
        '''
        new_data = []
        new_col = []
        for col in cols:
            new_col.append(self.column_names[col])
        for row in self.data:
            new_row = []
            for col in cols:
                new_row.append(row[col])
            new_data.append(new_row)
        new_table = MyPyTable(column_names=new_col, data=new_data)
        return new_table

    def convert_row_to_binary(self, col_name, base_val="0"):
        """Function to convert a row to binary based on if it does or does not have a value

        args:
            col_name: column to convert
            base_val: base value. Convert every other value
        """
        data = self.data
        col_idx = self.column_names.index(col_name)

        for row in data:
            if row[col_idx] == base_val:
                row[col_idx] = 0
            else:
                row[col_idx] = 1
        
        self.data = data
    
    def combine_boolean_rows(self, col1_name, col2_name):
        '''
        args:
            col1_name, col2_name (str): name of the two cols that should
                be merged
        returns:
            a new pytable!
        '''
        index1 = self.column_names.index(col1_name)
        index2 = self.column_names.index(col2_name)
        for row in self.data:
            if row[index1] or row[index2]:
                row[index1] = True
            else:
                row[index1] = False
        cols_to_keep = []
        for i in range(len(self.column_names)):
            if i is not index2:
                cols_to_keep.append(i)
        return self.keep_only_these_cols(cols_to_keep)

def main():
    """
        main for testing
    """
    header = ["CarName", "ModelYear", "MSRP"]
    msrp_table = [
        ["ford pinto", 75, "2769"],
        ["toyota corolla", 75, "2711"],
        ["ford pinto", 76, "3025"],
        ["toyota corolla", 77, "2789"],
    ]
    mytab = MyPyTable(header, msrp_table)
    mytab.pretty_print()
    print("shape (4,3): ", str(mytab.get_shape()))
    emptytab = MyPyTable()
    emptytab.pretty_print()
    print("shape (0,0): ", str(emptytab.get_shape()))
    print("get_col:", mytab.get_column("MSRP"))
    print(type(mytab.data[0][2]))
    mytab.convert_to_numeric()
    print(type(mytab.data[0][2]))
    print(mytab.find_duplicates(["CarName"]))
    mytab.drop_rows([3, 0])
    mytab.pretty_print()
    filename = os.path.join("test", "dummy.csv")
    table = MyPyTable().load_from_file(filename)
    table.pretty_print()
    outfilename = os.path.join("test", "car_out.csv")
    mytab.save_to_file(outfilename)
    outfilename = os.path.join("test", "dumm_out.csv")
    table.save_to_file(outfilename)


if __name__ == "__main__":
    main()
