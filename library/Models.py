import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, r2_score, mean_squared_error, confusion_matrix
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC

class models:
    def __init__(self, df):
        self.dataframe = df


    def create_folds(self, target_col, splits):
        self.dataframe["kfold"] = -1
        self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)
        kf = KFold(n_splits=splits, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe, y=self.dataframe[target_col].values)):
            self.dataframe.loc[val_idx, "kfold"] = fold
        self.dataframe.to_csv("current_df.csv", index=False)
        FOLD_MAPPING = {}
        for i in range(0, splits):
            temp_lst = []
            for j in range(0, splits):
                if j != i:
                    temp_lst.append(j)
            FOLD_MAPPING[i] = temp_lst
        return FOLD_MAPPING
        

    def split(self, FOLD_MAPPING, FOLD, target_col):
        train_df = self.dataframe[self.dataframe.kfold.isin(FOLD_MAPPING[FOLD])]
        valid_df = self.dataframe[self.dataframe.kfold == FOLD]
        ytrain = train_df[target_col].values
        yvalid = valid_df[target_col].values
        train_df.drop(["kfold", target_col], axis=1, inplace=True)
        valid_df.drop(["kfold", target_col], axis=1, inplace=True)
        valid_df = valid_df[train_df.columns]
        return train_df, valid_df, ytrain, yvalid


    def select_model(self, model_name, args):
        if model_name == "Linear_Regression":
            return LinearRegression()
        if model_name == "Logistic_Regression":
            return LogisticRegression(max_iter=args[0], random_state=42)
        elif model_name == "Decision_Tree_Classifier":      
            return DecisionTreeClassifier(random_state=42, criterion=args[0], max_depth=args[1])
        elif model_name == "Decision_Tree_Regressor":      
            return DecisionTreeRegressor(random_state=42, criterion=args[0], max_depth=args[1])
        elif model_name == "KNN_Classifier":
            return KNeighborsClassifier(n_neighbors=args[0], p=args[1])
        elif model_name == "KNN_Regressor":
            return KNeighborsRegressor(n_neighbors=args[0], p=args[1])
        elif model_name == "Naive_Bayes":
            return GaussianNB()
        elif model_name == "Adaboost_Classifier":
            return AdaBoostClassifier(n_estimators=args[0], learning_rate=args[1], random_state=42)
        elif model_name == "Adaboost_Regressor":
            return AdaBoostRegressor(n_estimators=args[0], learning_rate=args[1], loss=args[2], random_state=42)
        elif model_name == "Gradient_Boost_Classifier":
            return GradientBoostingClassifier(n_estimators=args[0], learning_rate=args[1], loss=args[2], random_state=42)
        elif model_name == "Gradient_Boost_Regressor":
            return GradientBoostingRegressor(n_estimators=args[0], learning_rate=args[1], loss=args[2], random_state=42)
        elif model_name == "SVM":
            return SVC(kernel=args[0], gamma=args[1], random_state=42)
        elif model_name == "Random_Forest_Classifier":
            return RandomForestClassifier(n_estimators=args[0], criterion=args[1], max_depth=args[2], random_state=42)
        elif model_name == "Random_Forest_Regressor":
            return RandomForestRegressor(n_estimators=args[0], criterion=args[1], max_depth=args[2], random_state=42)
        
    
    def classification_metrics(self, yvalid, preds, acc_ls, f1_ls, precision_ls, recall_ls, auc_l):
        acc = accuracy_score(yvalid, preds)
        f1 = f1_score(yvalid, preds)
        precision = precision_score(yvalid, preds)
        recall = recall_score(yvalid, preds)
        auc = roc_auc_score(yvalid, preds)
        acc_ls.append(acc)
        f1_ls.append(f1)
        precision_ls.append(precision)
        recall_ls.append(recall)
        auc_l.append(auc)
        return acc_ls, f1_ls, precision_ls, recall_ls, auc_l


    def regression_metrics(self, yvalid, preds, mse_ls, r2_ls):
        r2 = r2_score(yvalid, preds)
        mse = mean_squared_error(yvalid, preds)
        mse_ls.append(mse)
        r2_ls.append(r2)
        return mse_ls, r2_ls


    def train(self, target_col, splits, model_name, model_type, *args):
        FOLD_MAPPING = self.create_folds(target_col, splits)
        clf = self.select_model(model_name, args)
        score = 0
        error = 0
        acc_ls = [] 
        f1_ls = []
        precision_ls = []
        recall_ls = []
        auc_ls = []
        mse_ls = []
        r2_ls = []
        for FOLD in FOLD_MAPPING:
            print(f"################### FOLD {FOLD} #####################")
            train_df, valid_df, ytrain, yvalid = self.split(FOLD_MAPPING, FOLD, target_col)
            clf.fit(train_df, ytrain)
            preds = clf.predict(valid_df)
            if model_type == "classification":
                acc_ls, f1_ls, precision_ls, recall_ls, auc_ls = self.classification_metrics(yvalid, preds, acc_ls, f1_ls, precision_ls, recall_ls, auc_ls)
            else:
                mse_ls, r2_ls = self.regression_metrics(yvalid, preds, mse_ls, r2_ls)
        if model_type == "regression":
            dict = {
                "mse": sum(mse_ls) / len(mse_ls),
                "r2": sum(r2_ls) / len(r2_ls),
                # "coef_lst": clf.coef_,
                # "intercept": clf.intercept_
            }
        else:
            dict = {
                    "acc": sum(acc_ls) / len(acc_ls),
                    "f1": sum(f1_ls) / len(f1_ls),
                    "precision": sum(precision_ls) / len(precision_ls),
                    "recall": sum(recall_ls) / len(recall_ls),
                    "auc": sum(auc_ls) / len(auc_ls),
                    # "coef_lst": clf.coef_,
                    # "intercept": clf.intercept_, 
                }
        print (dict)
        return dict

if __name__ == "__main__":
    df = pd.read_csv("./current_df.csv")
    model = models(df)
    target_col = "Outcome"
    model.train(target_col, 5, "SVM", "regression", "linear", "scale")
