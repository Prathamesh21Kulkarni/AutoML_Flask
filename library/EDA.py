import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import json


class Basic_Info:
    def __init__(self, df):
        self.dataframe = df

    # Shows shape of dataset
    def print_shape(self):
        return self.dataframe.shape

    #  Shows column names in dataset
    def display_columns(self):
        return self.dataframe.columns

    # Shows max, min, 25%, 50%, 75%, mean, std column-wise
    def display_summary(self):
        return self.dataframe.describe()

    # Shows no. of missing value in every column
    def show_missing_values(self):
        return self.dataframe.isnull().sum()


class NumericDetails:
    def __init__(self, df) -> None:
        self.dataframe = df

    def getDetails(self):
        numColList = [
            col
            for col in self.dataframe.columns
            if str(self.dataframe[col].dtype).find("object") == -1
        ]
        numeric_details = []
        index = 0
        # ['n_distinct','n_unique','type','mean','n_missing','p_missing','std','variance','min','max','range','25%','50%','75%','iqr','kurtosis','skewness','chi_squared']
        for eachCol in numColList:
            finalDict = {}
            finalDict["column_name"] = eachCol
            finalDict["total_values"] = self.dataframe.shape[0]
            finalDict["n_distinct"] = str(self.dataframe[eachCol].nunique())
            # n_distinct and n_unique means the same right?
            finalDict["type"] = "Numeric"
            n_missing = self.dataframe[eachCol].isnull().sum()
            finalDict["n_missing"] = str(n_missing)
            per = n_missing / (int)(self.dataframe[eachCol].count()) * 100
            finalDict["p_missing"] = str(per)
            finalDict["kurtosis"] = str(self.dataframe[eachCol].kurtosis())
            finalDict["skewness"] = str(self.dataframe[eachCol].skew())
            temp_df = pd.DataFrame(finalDict, index=[index])
            numeric_details.append(temp_df)
            index += 1
        numeric_details = pd.concat(numeric_details)
        return numeric_details


class CategoricalDetails:
    def __init__(self, df) -> None:

        self.dataframe = df

    def getDetails(self):
        catColList = [
            col
            for col in self.dataframe.columns
            if str(self.dataframe[col].dtype).find("object") != -1
        ]
        categorical_details = []
        index = 0
        for col in catColList:
            finalDict = {}
            finalDict["column_name"] = col
            finalDict["total_values"] = self.dataframe.shape[0]
            finalDict["n_distinct"] = str(self.dataframe[col].nunique())
            finalDict["type"] = "Categorical"
            # finalDict["category_list"] = str(self.dataframe[col].unique())
            # finalDict["word_count"] = dict(self.dataframe[col].value_counts().to_dict())
            finalDict["n_missing"] = int(self.dataframe[col].isnull().sum())
            finalDict["p_missing"] = str(
                finalDict["n_missing"] / (int)(self.dataframe[col].count()) * 100
            )
            max_len = -1
            min_len = 1000
            for ele in self.dataframe[col].value_counts().to_dict().keys():
                if len(str(ele)) > max_len:
                    max_len = len(str(ele))
                if len(str(ele)) < min_len:
                    min_len = len(str(ele))
            finalDict["max_length"] = int(max_len)
            finalDict["min_length"] = int(min_len)
            finalDict["len_range"] = int(max_len - min_len)
            temp_df = pd.DataFrame(finalDict, index=[index])
            index += 1
            categorical_details.append(temp_df)
        if len(categorical_details) > 0:
            categorical_details = pd.concat(categorical_details)
        return categorical_details

    def get_head(self):
        return self.dataframe.head()


# Need to add the chart classes here as well


class Charts:
    def __init__(self) -> None:
        self.dataframe = pd.read_csv("current_df.csv")

    def line(self, colname1, colname2):
        labels = self.dataframe[colname1].to_list()
        values = self.dataframe[colname2].to_list()
        return labels, values

    # These return values would be then passed to the chart.js through flask

    def histogram(self, colname):
        labels = list(self.dataframe[colname].value_counts().to_dict().keys())
        values = list(self.dataframe[colname].value_counts().to_dict().values())
        return labels, values

    def scatter(self, colname1, colname2):
        keys = self.dataframe[colname1].values.tolist()
        values = self.dataframe[colname2].values.tolist()
        dict = {k: v for (k, v) in zip(keys, values)}
        return dict

    def pie(self, colname):
        # Pie should be applicable only for categorical values or some numeric values if they are representing some category(e.g. rating 1 2 3 4, etc) and we have kept here a restriction that the no of unique values must be < 40 so the there would be some meaning to the chart
        labels = list(self.dataframe[colname].unique())
        data = []
        for each_label in labels:
            data.append(
                self.dataframe[self.dataframe[colname] == each_label][colname].count()
            )
        return labels, data

    def get_charts_data(self):
        # num_col_list =  [
        #     col
        #     for col in self.dataframe.columns
        #     if str(self.dataframe[col].dtype).find("object") == -1
        # ]
        # cat_col_list = [col for col in self.dataframe.columns if str(self.dataframe[col].dtype).find("object") != -1]
        charts_dict = {}
        # num_bar_details = []
        hist_details = {}
        # cat_pie_details = []
        for each_col in self.dataframe.columns:
            labels, values = self.histogram(each_col)
            hist_details[each_col] = {
                "labels": labels,
                "values": values,
            }
            # hist_details.append(temp_dict)
            # temp_dict = {}
            # labels, data = self.pie(each_col)
            # temp_list = []
            # for i in range(len(labels)):
            #     temp_dict = {}
            #     temp_dict["label"] = labels[i]
            #     temp_dict["value"] = values[i]
            #     temp_list.append(json.dumps(temp_dict))
            # temp_dict['''"''' + each_col + '''"'''] = temp_list
            # cat_pie_details.append(json.dumps(temp_dict))
        # charts_dict = hist_details
        # charts_dict["pie"] = cat_pie_details
        charts_dict = json.dumps(hist_details)
        return charts_dict


def perform_EDA():
    df = pd.read_csv("current_df.csv")
    basic_info = Basic_Info(df)
    num_details = NumericDetails(df)
    cat_details = CategoricalDetails(df)
    n = len(cat_details.getDetails())
    if  n > 0:
        eda_dict = {
            "shape": str(basic_info.print_shape()),
            "column_names": list(basic_info.display_columns()),
            "summary": json.dumps(basic_info.display_summary().to_dict()).replace('"', "'"),
            "missing_values": basic_info.show_missing_values().to_json(),
            "num_detail_table": num_details.getDetails().to_json(),
            "cat_detail_table": cat_details.getDetails().to_json(),
            "df_head": cat_details.get_head().to_json(),
        }
    else :
        eda_dict = {
            "shape": str(basic_info.print_shape()),
            "column_names": list(basic_info.display_columns()),
            "summary": json.dumps(basic_info.display_summary().to_dict()).replace('"', "'"),
            "missing_values": basic_info.show_missing_values().to_json(),
            "num_detail_table": num_details.getDetails().to_json(),
            "cat_detail_table":{},
            "df_head": cat_details.get_head().to_json(),
        }

    # print(eda_dict["summary"].replace('"',"'"))
    return eda_dict


# def Charts_data():
#     # df = pd.read_csv("current_df.csv")
#     charts = Charts()
#     return charts.histogram("Attribute")
