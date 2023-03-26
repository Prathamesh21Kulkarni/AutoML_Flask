from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import pandas as pd
from library.EDA import perform_EDA
from library.EDA import Charts
import json
import os
from library.Cleaning import Missing_Values, Outliers
from library.Preprocessing import Categorical_Encoding
from library.Feature_Selection import feature_selection
from library.Models import models
import numpy as np

app = Flask(__name__)

def fill_missing(col_name, option):
    df = pd.read_csv('current_df.csv')
    MS = Missing_Values(df)
    if option == "Fill_with_0":
        MS.fill_0(col_name)
    elif option == "Fill_with_mean":
        MS.fill_mean(col_name)
    elif option == "Fill_with_forward_value":
        MS.fill_forward(col_name)
    elif option == "Fill_with_backward_value":
        MS.fill_backward(col_name)
    elif option == "Fill_with_mode":
        MS.fill_frequency(col_name)
@app.route("/")
def upload_file():
    return render_template("index.html")


# @app.route("/uploader", methods=["GET", "POST"])
# def upload_file():
#     if request.method == "POST": nhk  
#         f = request.files["file"]
#         f.save(secure_filename(f.filename))
#         return "file uploaded successfully"

@app.route("/feature_selection")
def reduce_features():
    return render_template("feature.html")

@app.route("/model_training_and_testing")
def train_test():
    return render_template("training_testing.html")
@app.route("/model_evaluation")
def evaluate():
    return render_template("model_evaluation.html")
@app.route('/feature_engineering_missing_filled', methods=['POST'])
def process_form():
    if button == 'ok':
        option_lst = request.form['option'].split("@")
        button = request.form['action']
        option = option_lst[0]
        col_name = option_lst[1]
        fill_missing(col_name, option)
    df = pd.read_csv("current_df.csv", encoding='utf8')
    eda_dict = perform_EDA()
    df = json.dumps(df.to_dict()).replace('"',"'")
    charts_dict = Charts().get_charts_data()
    return render_template("dashboard/index.html", eda_dict=eda_dict, charts_dict = charts_dict)

def encode(col_name, option, df):
    CE = Categorical_Encoding(df)
    if option == "Label_encoding":
        CE.label_encoding(col_name)
    elif option == "One_hot_encoding":
        CE.onehot_encoding(col_name)
    elif option == "Label_Binarization":
        CE.label_binarization(col_name)


@app.route('/feature_engineering_encoded', methods=['POST'])
def process_form1():
    option_lst = request.form['option'].split("@")
    button = request.form['action']
    print(option_lst)
    option = option_lst[0]
    col_name = option_lst[1]
    df = pd.read_csv("current_df.csv", encoding='utf8')
    if button == 'ok':
        encode(col_name, option, df)
    eda_dict = perform_EDA()
    df = json.dumps(df.to_dict()).replace('"',"'")
    charts_dict = Charts().get_charts_data()
    return render_template("dashboard/index.html", eda_dict=eda_dict, charts_dict = charts_dict)

def handle_outliers(df, col_name, method_to_handle):
    O = Outliers(df)
    technique_to_detect = "Percentile"
    outlier_df, outlier_index = O.detect_outliers(col_name, technique_to_detect)
    print(outlier_df)
    O.fill_outliers(outlier_index, col_name, method_to_handle)

@app.route('/feature_engineering_handle_outliers', methods=['POST'])
def process_form2():
    df = pd.read_csv("current_df.csv", encoding='utf8')
    button = request.form['action']
    if button == 'ok':
        option_lst = request.form['option'].split("@")
        print(option_lst)
        method_to_handle = option_lst[0]
        col_name = option_lst[1]
        handle_outliers(df, col_name, method_to_handle)
    eda_dict = perform_EDA()
    df = json.dumps(df.to_dict()).replace('"',"'")
    charts_dict = Charts().get_charts_data()
    return render_template("dashboard/index.html", eda_dict=eda_dict, charts_dict = charts_dict)

@app.route("/feature_engineering", methods=["GET", "POST"])
def upload_files():
    if request.method == "POST":
        f = request.files["file"]
        f.save(secure_filename("current_df.csv"))
        eda_dict = perform_EDA()
        df = pd.read_csv("current_df.csv", encoding='utf8')
        # print(df.head())
        df = json.dumps(df.to_dict()).replace('"',"'")
        print(df[0])
        
        ch = Charts()
        charts_dict = ch.get_charts_data()
    return render_template("dashboard/index.html", eda_dict=eda_dict, charts_dict = charts_dict)

@app.route('/feature_selection_lasso', methods=['POST'])
def process_form3():
    df = pd.read_csv("current_df.csv", encoding='utf8')
    button = request.form['action']
    if button == 'ok':
        target_col = request.form['target']
        print("###########")
        print(target_col)
        print("###########")
        y = df[target_col].values
        df.drop([target_col], inplace=True, axis=1)
        x = []
        for i in range(0, df.shape[0]):
            x.append(df.iloc[i].values.tolist())
        fe = feature_selection(x, y)
        lasso_coefficients = fe.lasso()
        lasso_coefficients.sort()
        dict = {}
        i = 0
        for col in df.columns:
            dict[col] = lasso_coefficients[i]
            i += 1
        df = json.dumps(df.to_dict()).replace('"',"'")
    return render_template("feature.html", dict = dict)
    
@app.route('/feature_selection_ridge', methods=['POST'])
def process_form4():
    df = pd.read_csv("current_df.csv", encoding='utf8')
    button = request.form['action']
    if button == 'ok':
        target_col = request.form['target']
        print("###########")
        print(target_col)
        print("###########")
        y = df[target_col].values
        df.drop([target_col], inplace=True, axis=1)
        x = []
        for i in range(0, df.shape[0]):
            x.append(df.iloc[i].values.tolist())
        fe = feature_selection(x, y)
        ridge_coefficients = fe.ridge()
        ridge_coefficients.sort()
        # ridge_coefficients = np.flip(ridge_coefficients)
        dict = {}
        i = 0
        for col in df.columns:
            dict[col] = ridge_coefficients[i]
            i += 1
        df = json.dumps(df.to_dict()).replace('"',"'")
    return render_template("feature.html", dict = dict)


@app.route('/feature_selection_rfe', methods=['POST'])
def process_form5():
    df = pd.read_csv("current_df.csv", encoding='utf8')
    button = request.form['action']
    if button == 'ok':
        target_col = request.form['target']
        print("###########")
        print(target_col)
        print("###########")
        y = df[target_col].values
        df.drop([target_col], inplace=True, axis=1)
        x = []
        for i in range(0, df.shape[0]):
            x.append(df.iloc[i].values.tolist())
        fe = feature_selection(x, y)
        rfe_coefficients = fe.random_forest_feature_selection()
        rfe_coefficients.sort()
        dict = {}
        i = 0
        for col in df.columns:
            dict[col] = rfe_coefficients[i]
            i += 1
        df = json.dumps(df.to_dict()).replace('"',"'")
    return render_template("feature.html",dict = dict)
@app.route('/model_training_and_testing', methods=['POST'])
def process_form6():
    df = pd.read_csv("current_df.csv")
    button = request.form['action']
    if button == 'ok':
        col_ = request.form['column_data']
        col_dict = json.loads(col_)
        print("###########")
        print(col_dict)
        print("###########")
        cols_to_be_included = []
        for key in col_dict.keys():
            cols_to_be_included.append(key)
            # print(key)
        cols_to_be_included.pop(0)
        cols_to_be_included.append(col_dict["TargetColumn"])
        df = df[cols_to_be_included]
        df.to_csv("current_df.csv", index=False)
        dict = {}
    return render_template("feature.html",dict=dict)
        # return render_template("dashboard/index.html", eda_dict=eda_dict)
@app.route('/model_training_linearReg', methods=['POST'])
def process_form7():
    df = pd.read_csv("current_df.csv", encoding='utf8')
    button = request.form['action']
    if button == 'ok':
        target_col = request.form['target']
        splits = request.form['splits']
        lr = models(df)
        dict = lr.linear_regression(target_col, int(splits))
    return render_template("training_testing.html",dict = dict)

if os.path.isfile("current_df.csv"):
    df = pd.read_csv("current_df.csv")
else :
    df = pd.DataFrame()

# for col in df.columns:
# @app.route("/feature_engineering/", method=["GET"])
# def column_details():
#         col_dict = {
#             "type":"Categorical",
#             "value_counts": 2430,
#             "missing_values": 789
#         }
#         return render_template("index.html", col_dict = col_dict)



if __name__ == "__main__":
    app.run(debug=True, port=5001)
