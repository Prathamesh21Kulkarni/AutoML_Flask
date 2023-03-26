from tkinter import FIRST
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


class Missing_Values:
    def __init__(self, df) -> None:
        self.dataframe = df

    # https://scikit-learn.org/stable/modules/impute.html#univariate-feature-imputation
    # Fill all missing values with 0
    def fill_0(self, col):
        self.dataframe[col] = self.dataframe[col].fillna(0)
        self.dataframe.to_csv("current_df.csv", index=False)

    # Fill all missing values with mean (must be done column-wise)
    def fill_mean(self, col):
        if self.dataframe[col].dtype != object:
            df_num = pd.DataFrame(self.dataframe[col])
            mean = self.dataframe[col].mean()
            self.dataframe[col].fillna(mean, inplace=True)
            self.dataframe.to_csv("current_df.csv", index=False)
            # print(self.dataframe[col])
        else:
            print("Cannot perform mean operation on column -> " + col)

    # Fill all missing values with value of next row (must be done column-wise)
    def fill_forward(self, col):
        # print("\nReplacing NaNs with the value from the next row :")
        self.dataframe[col].fillna(method="bfill", inplace=True)
        self.dataframe.to_csv("current_df.csv", index=False)

    # Fill all missing values with value of previous row (must be done column-wise)
    def fill_backward(self, col):
        # print("\nReplacing NaNs with the value from the previous row :")
        self.dataframe[col].fillna(method="pad", inplace=True)
        self.dataframe.to_csv("current_df.csv", index=False)

    # Fill all missing values with mode (must be done column-wise)
    def fill_frequency(self, col):
        print(self.dataframe[col].mode())
        self.dataframe[col].fillna(self.dataframe[col].mode()[0], inplace=True)
        self.dataframe.to_csv("current_df.csv", index=False)


class Outliers:
    def __init__(self, df) -> None:
        self.dataframe = df
    max_threshold, min_threshold = 0, 0
    # check = True  # So that the program doesn't go in infinite loop

    # Need to check this
    def show_boxplot(self, column):
        sns.boxplot(self.dataframe[column])
    # -------------------------------------------------------------------------
        # For dealing with outliers there are 3 methods:
        # Trimming/removing the outlier
        # Quantile based flooring and capping
        # Mean/Median imputation
    # -------------------------------------------------------------------------

    def trim(self, rows):
        self.dataframe = self.dataframe.drop(labels=rows, inplace=True)
        self.dataframe.to_csv("current_df.csv", index=False)

    def cap(self, outlier_index, colname, mean, median):
        # self.show_boxplot(colname)
        if(mean == True):
            Mean = self.dataframe[colname].mean()
            for index in outlier_index:
                self.dataframe.loc[index, colname] = Mean
        elif(median == True):
            Median = self.dataframe[colname].median()
            for index in outlier_index:
                self.dataframe.loc[index, colname] = Median
        else: # cap to min or max
            for i in outlier_index:
                if(self.dataframe.loc[i][colname] > self.max_threshold):
                    self.dataframe.loc[i, colname] = self.max_threshold
                elif(self.dataframe.loc[i][colname] < self.min_threshold):
                    self.dataframe.loc[i, colname] = self.min_threshold
        self.dataframe.to_csv("current_df.csv", index=False)

    def detect_outliers(self, colname, technique_to_detect):
        rows = []
        if(technique_to_detect == "IQR"):
            print(technique_to_detect)
            Q1, Q3 = self.dataframe[colname].quantile([0.25, 0.75])
            self.max_threshold = Q3 + 1.5*(Q3-Q1)
            self.min_threshold = Q1 - 1.5*(Q3-Q1)
            rows_to_be_dropped = self.dataframe[(self.dataframe[colname] < self.min_threshold) | (self.dataframe[colname] > self.max_threshold)]
            outlier_df = rows_to_be_dropped
            outlier_index = outlier_df.index.values
            if len(outlier_index) == 0:
                print("No outliers present")
        else:
            self.min_threshold, self.max_threshold = self.dataframe[colname].quantile([0.001, 0.99])
            rows_to_be_dropped = self.dataframe[(self.dataframe[colname] < self.min_threshold) | (self.dataframe[colname] > self.max_threshold)].index
            outlier_index = rows_to_be_dropped.tolist()
            outlier_df = self.dataframe.loc[outlier_index]
            outlier_index = outlier_df.index.values
            if len(outlier_index) == 0:
                print("No outliers present")
        return outlier_df, outlier_index

    def fill_outliers(self, outlier_index, colname, method_to_handle):
        if(method_to_handle == "Trim"):
            self.trim(outlier_index)
        elif(method_to_handle == "Cap_min_max"):
            self.cap(outlier_index, colname, False, False)
        elif(method_to_handle == "Cap_mean"):
            self.cap(outlier_index, colname, True, False)
        elif(method_to_handle == "Cap_median"):
            self.cap(outlier_index, colname, False, True)


class Duplicates:
    def __init__(self, df) -> None:
        self.dataframe = df

    def duplicates_removal(self, within_col, col, row):
        # There are 3 types of duplicates that we can remove
        # 1) Duplicates values within a column i.e. duplicate values within a column
        # 2) Duplicates columns e.g. if 2 columns are same
        # 3) Duplicate rows within the dataset

        # ------------About .drop_duplicates() method ------------#
        # Syntax: DataFrame.drop_duplicates(subset=None, keep=’first’, inplace=False)
        # Parameters:
        # subset: Subset takes a column or list of column label. It’s default value is none. After passing columns, it will consider them only for duplicates.
        # keep: keep is to control how to consider duplicate value. It has only three distinct value and default is ‘first’.
        # If ‘first‘, it considers first value as unique and rest of the same values as duplicate.
        # If ‘last‘, it considers last value as unique and rest of the same values as duplicate.
        # If False, it consider all of the same values as duplicates
        # inplace: Boolean values, removes rows with duplicates if True.
        print(self.dataframe.shape)
        if within_col == True:
            # Here we need to ask user which col
            col = "Lv 50 HP"
            self.dataframe.drop_duplicates(subset=col, keep=FIRST, inplace=True)
        elif col == True:
            # For demo purpose we are adding a duplicate col
            # print(self.dataframe.shape)
            # self.dataframe['new_col'] = self.dataframe["Lv 50 HP"]
            # print(self.dataframe.shape)
            # self.dataframe.T.drop_duplicates().T
            cols = self.dataframe.columns
            print(cols)
            cols_to_remove = []
            for i in range(0, len(cols) - 1):
                for j in range(i + 1, len(cols)):
                    if self.dataframe[cols[i]].equals(self.dataframe[cols[j]]):
                        print(cols[j])
                        cols_to_remove.append(cols[j])
            self.dataframe.drop(cols_to_remove, inplace=True, axis=1)

            # print(self.dataframe.shape)
            # self.dataframe.drop_duplicates(inplace=True)
        elif row == True:
            self.dataframe.drop_duplicates(keep=FIRST, inplace=True)
        print(self.dataframe.shape)
