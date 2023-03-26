from sklearn.linear_model import LinearRegression
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, r2_score, mean_squared_error

class models:
    def __init__(self, df):
        self.dataframe = df

    def create_folds(self, target_col, splits):
        self.dataframe["kfold"] = -1
        self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)
        kf = KFold(n_splits=splits, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe, y=self.dataframe[target_col].values)):
            print(len(train_idx), len(val_idx))
            self.dataframe.loc[val_idx, "kfold"] = fold
        self.dataframe.to_csv("current_df.csv", index=False)

        FOLD_MAPPING = {}
        for i in range(0, splits):
            temp_lst = []
            for j in range(0, splits):
                if j != i:
                    temp_lst.append(j)
            FOLD_MAPPING[i] = temp_lst
        print(FOLD_MAPPING)
        return FOLD_MAPPING
        

    def linear_regression(self, target_col, splits):
        FOLD_MAPPING = self.create_folds(target_col, splits)
        score = 0
        error = 0
        fold_results = []
        clf = LinearRegression()

        for FOLD in FOLD_MAPPING:
            print(f"################### FOLD {FOLD} #####################")
            train_df = self.dataframe[self.dataframe.kfold.isin(FOLD_MAPPING[FOLD])]
            valid_df = self.dataframe[self.dataframe.kfold == FOLD]
            ytrain = train_df[target_col].values
            yvalid = valid_df[target_col].values

            train_df.drop(["kfold", target_col], axis=1, inplace=True)
            valid_df.drop(["kfold", target_col], axis=1, inplace=True)
            valid_df = valid_df[train_df.columns]

            clf.fit(train_df, ytrain)
            preds = clf.predict(valid_df)
            # print(preds)
            # print(yvalid)
            r2 = r2_score(yvalid, preds)
            fold_results.append(r2)
            mse = mean_squared_error(yvalid, preds)
            score += r2
            error += mse
        print(f"Final R2 Score = {(score/splits)*100}")
        print(f"Final Error = {(error/splits)}")
        print(clf.coef_)
        print(clf.intercept_)
        print(fold_results)
        dict = {};
        dict["score"] = (score/splits)*100
        dict["error"] = (error/splits)
        dict["coef"] = clf.coef_.tolist()
        dict["intercept"] = clf.intercept_
        dict["fold_results"] = fold_results
        return dict


    def logistic_regression(x, y):
        pass

if __name__ == "__main__":
    df = pd.read_csv("../current_df.csv")
    model = models(df)
    target_col = "Selling_Price"
    model.linear_regression(target_col, 5)
