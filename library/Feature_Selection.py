from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import seaborn as sns

# Here we are using the embedded technique for feature selection


class feature_selection:

    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

    def splitting_into_train_and_test(self):
        from sklearn.model_selection import train_test_split
        split_size = 0.3  # we can take this from user
        x_train, x_test, y_train, y_test = train_test_split(
            self.x, self.y, test_size=split_size, random_state=17)
        return x_train, x_test, y_train, y_test

    def get_normal_regression(self, x_train, x_test, y_train, y_test):
        from sklearn.linear_model import LinearRegression
        # Here we can ask user which model to choose, like linear, logistic,svm,etc any model
        lr = LinearRegression()
        lr.fit(x_train, y_train)
        train_score_lr = lr.score(x_train, y_train)
        test_score_lr = lr.score(x_test, y_test)
        print("The train score for linear model is {}".format(train_score_lr))
        print("The test score for linear model is {}".format(test_score_lr))
    # We can take alpha from user and alpha must be a positive number starting from 0.But we would also be suggesting the best value of alpha with by running the model for different values of alpha with get_best_alpha method

    def get_best_alpha(self, model, x_train, y_train):
        # print('hi')
        from sklearn.model_selection import GridSearchCV
        parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-3,
                                1e-2, 1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 100]}
        if(model == "lasso"):
            from sklearn.linear_model import Lasso
            lasso = Lasso()
            lasso_regressor = GridSearchCV(
                lasso, parameters, scoring='neg_mean_squared_error', cv=5)
            lasso_regressor.fit(x_train, y_train)
            print(lasso_regressor.best_params_)
            print(lasso_regressor.best_score_)
            return lasso_regressor.best_params_
        elif(model == "ridge"):
            from sklearn.linear_model import Ridge
            ridge = Ridge()
            ridge_regressor = GridSearchCV(
                ridge, parameters, scoring='neg_mean_squared_error', cv=5)
            ridge_regressor.fit(x_train, y_train)
            print(ridge_regressor.best_params_)
            print(ridge_regressor.best_score_)
            return ridge_regressor.best_params_

    def lasso(self):
        from sklearn.linear_model import Lasso
        x_train, x_test, y_train, y_test = self.splitting_into_train_and_test()
        self.get_normal_regression(x_train, x_test, y_train, y_test)
        # We can take alpha from user and alpha must be a positive number starting from 0.But we would also be suggesting the best value of alpha with by running the model for different values of alpha with get_best_alpha method
        alpha = self.get_best_alpha("lasso", x_train, y_train)
        print("ALPHA = " + str(alpha['alpha']))
        lasso = Lasso(alpha['alpha'])
        lasso.fit(x_train, y_train)
        print("Testing score")
        print(lasso.score(x_test, y_test))
        print("Training score")
        print(lasso.score(x_train, y_train))
        print("Coef of each columns")
        print(lasso.coef_)
        return lasso.coef_
        # Selection of coefficients based on threshold values is remaining

    def ridge(self):
        from sklearn.linear_model import Ridge
        # from sklearn.feature_selection import SelectFromModel
        print("Ridge")
        x_train, x_test, y_train, y_test = self.splitting_into_train_and_test()
        self.get_normal_regression(x_train, x_test, y_train, y_test)
        # We can take alpha from user and alpha must be a positive number starting from 0.But we would also be suggesting the best value of alpha with by running the model for different values of alpha with get_best_alpha method
        alpha = self.get_best_alpha("ridge", x_train, y_train)
        print("ALPHA = " + str(alpha['alpha']))

        ridge = Ridge(alpha['alpha'])
        # cv means no of cross-validations. so it could also be taken by user

        ridge.fit(x_train, y_train)
        print("Testing score")
        print(ridge.score(x_test, y_test))
        print("Training score")
        print(ridge.score(x_train, y_train))
        print("Coef of each columns")
        return ridge.coef_
        # Selection of coefficients based on threshold values is remaining

    def random_forest_feature_selection(self):
        # Adding scaling process here, just for testing purpose, later have to remove it
        from sklearn.preprocessing import StandardScaler
        x_train, x_test, y_train, y_test = self.splitting_into_train_and_test()
        sc = StandardScaler()
        x_train_std = sc.fit_transform(x_train)
        print("x_train_std shape : " + str(x_train_std.shape))
        x_test_std = sc.transform(x_test)
        print("x_test_std shape : " + str(x_test_std.shape))
        from sklearn.ensemble import RandomForestClassifier
        forest = RandomForestClassifier(n_estimators=500,random_state=1)  # We can ask user about the no of estimators he want i.e. no of decision trees he want in the random forest classifier
        forest.fit(x_train_std, y_train)
        # this returns the coefficient of each feature
        return forest.feature_importances_

    def rfe(self):
        from sklearn.preprocessing import StandardScaler
        
        x_train, x_test, y_train, y_test = self.splitting_into_train_and_test()
        sc = StandardScaler()
        x_train_std = sc.fit_transform(x_train)
        print("x_train_std shape : " + str(x_train_std.shape))
        x_test_std = sc.transform(x_test)
        print("x_test_std shape : " + str(x_test_std.shape))
        from sklearn.linear_model import LogisticRegression
        from sklearn.feature_selection import RFE
        lr = LogisticRegression(solver='liblinear', random_state=123) # Ask user which model he wants to choose
        print("Im here")
        from sklearn.model_selection import GridSearchCV
        rfe = RFE(estimator=lr, step=1) # Use here random forest
        # print(rfe.get_params().keys())
        parameters = {'n_features_to_select': range(1, 10)}
        grid = GridSearchCV(RFE(estimator=lr, step=1),param_grid=parameters, cv=2)
        grid.fit(x_train_std, y_train)
        print('Best params:', grid.best_params_)
        features=list(pd.DataFrame(self.x).columns[grid.best_estimator_.support_])
        print('Best feature names:', features)
        print('Best accuracy:', grid.best_score_)


        
if __name__ == "__main__":
    df = pd.read_csv("../current_df.csv")
    target_col = "target"
    y = df[target_col].values
    df.drop([target_col], inplace=True, axis=1)
    x = []
    for i in range(0, df.shape[0]):
        x.append(df.iloc[i].values.tolist())

    fe = feature_selection(x, y)
    print(fe.lasso())
