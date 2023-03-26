# For Normalization and standardization
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import scipy.stats as stats

class date_time_conversion:
    def __init__(self,df) -> None:
        self.dataframe = df
    def process_date_time(self, colname):
        try:
            print(self.dataframe[colname].dtype)
            self.dataframe[str(colname + '_')] =  pd.to_datetime(self.dataframe[colname])
            print(self.dataframe[str(colname + '_')].dtype)
            print(self.dataframe.head())
        except:
            print("Specified col is not suitable for converting into datetime object")

class chi_squared:
    def __init__(self, df) -> None:
        self.dataframe = df

    def get_values(self, col1, col2):  # Here validate that col1 and col2 are categorical
        if str(self.dataframe[col1].dtype).find('object') == -1:
            print(col1 + " is not categorical. Select a categorical col")
            return
        elif str(self.dataframe[col2].dtype).find('object') == -1:
            print(col2 + " is not categorical. Select a categorical col")
            return
        dataset_table = pd.crosstab(self.dataframe[col1], self.dataframe[col2])
        Observed_Values = dataset_table.values
        val = stats.chi2_contingency(dataset_table)
        Expected_Values = val[3]
        no_of_rows = len(dataset_table.iloc[0:2, 0])
        no_of_columns = len(dataset_table.iloc[0, 0:2])
        ddof = (no_of_rows-1)*(no_of_columns-1)
        alpha = 0.05
        from scipy.stats import chi2
        chi_square = sum(
            [(o-e)**2./e for o, e in zip(Observed_Values, Expected_Values)])
        chi_square_statistic = chi_square[0]+chi_square[1]
        print("chi-square statistic:-", chi_square_statistic)
        p_value = 1-chi2.cdf(x=chi_square_statistic, df=ddof)
        critical_value=chi2.ppf(q=1-alpha,df=ddof)
        print('p-value:', p_value)
        print('Significance level: ', alpha)
        print('Degree of Freedom: ', ddof)
        print('p-value:', p_value)
        if chi_square_statistic>=critical_value:
            print("Reject H0,There is a relationship between 2 categorical variables")
        else:
            print("Retain H0,There is no relationship between 2 categorical variables")
            
        if p_value<=alpha:
            print("Reject H0,There is a relationship between 2 categorical variables")
        else:
            print("Retain H0,There is no relationship between 2 categorical variables")

class Normalization:
    # To be done by Kushal
    def __init__(self, df) -> None:
        self.dataframe = df

    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

    def standard_scaler(self):
        numColList = [col for col in self.dataframe.columns if str(
            self.dataframe[col].dtype).find('object') == -1]
        for eachCol in numColList:
            scaler = StandardScaler()
            print(scaler.fit(self.dataframe[eachCol]))
            print(scaler.mean_)
            print(scaler.transform(self.dataframe[eachCol]))
        pass

    def robust_scaler(self):
        transformer = RobustScaler()
        print(transformer.fit(self.dataframe))
        print(transformer.transform(self.dataframe))
        pass


class Categorical_Encoding:
    # To be done by Prathamesh
    def __init__(self, df) -> None:
        self.dataframe = df

    def handling_numeric_and_catgorical(self, colName):
        dt = self.dataframe.dtypes[str(colName)]
        if(str(dt).find('object')):
            return True
        else:
            # Here we can also check a parameter like if no of unique attributes is less than half or something like this
            if(len(self.dataframe[str(colName)].unique()) < len(self.dataframe[str(colName)])):
                return True
            else:
                return False
    #  https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html

    def label_encoding(self, colName):
        if(self.handling_numeric_and_catgorical(colName)):
            from sklearn import preprocessing
            label_encoder = preprocessing.LabelEncoder()
            self.dataframe[str(colName)] = label_encoder.fit_transform(self.dataframe[str(colName)])
            print("Data encoded successfully")
            print(self.dataframe.head())
        else:
            print("Given column is not suitable for label encoding")
        self.dataframe.to_csv("current_df.csv", index=False)

    def _label_encoding(self):  # For all columns
        cols = self.dataframe.columns
        for each_col in cols:
            self.label_encoding(each_col)
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

    def onehot_encoding(self, col):
        if(self.handling_numeric_and_catgorical(col)):
            uniques = len(self.dataframe[str(col)].unique())
            # self.dataframe = pd.DataFrame()
            # print(uniques)
            dumnies = pd.get_dummies(self.dataframe[col])
            print(dumnies.columns)
            dumnies = dumnies.drop(dumnies.columns[0], axis=1)
            print(dumnies.columns)
            for i in dumnies.columns:
                new_name = str(col) + "_" + str(i)
                dumnies[new_name] = dumnies[i]
                dumnies.drop([i], axis=1, inplace=True)
            self.dataframe = pd.concat([self.dataframe, dumnies], axis=1)
            self.dataframe.drop([col], axis=1, inplace=True)
            print(self.dataframe.head())
        else:
            print("Given column is not suitable for one hot encoding")
        self.dataframe.to_csv("current_df.csv", index=False)

    def _onehot_encoding(self):  # For all columns
        cols = self.dataframe.columns
        for each_col in cols:
            self.onehot_encoding(each_col)
    # Note : The below functions would need exception handling mechanism while calling

    def label_binarization(self, col):
        if(self.handling_numeric_and_catgorical(col)):
            from sklearn import preprocessing
            lbl = preprocessing.LabelBinarizer()
            lbl.fit(self.dataframe[col].values)
            val = lbl.transform(self.dataframe[col].values)
            self.dataframe = self.dataframe.drop(col, axis=1)
            for j in range(val.shape[1]):
                new_col_name = col + f"__bin_{j}"
                self.dataframe[new_col_name] = val[:, j]
            # self.binary_encoders[col] = lbl
        else:
            print("Given column is not suitable for one hot encoding")
        self.dataframe.to_csv("current_df.csv", index=False)

    def _label_binarization(self):  # For all columns
        cols = self.dataframe.columns
        for each_col in cols:
            self.label_binarization(each_col)
