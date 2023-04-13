# Import libraries
import os
import sys
import pickle
from datetime import datetime
import numpy as np
import mlflow
import pandas as pd
import whylogs as why
from prefect import flow, task
from whylogs.app import Session
from whylogs.proto import ModelType
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from whylogs.app.writers import WhyLabsWriter
from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import PorterStemmer

from preprocessing import remove_pattern

ps = PorterStemmer()

mlflow.set_tracking_uri("sqlite:///mydb.sqlite")
EXPERIMENT_NAME = "sentiment-analysis"
mlflow.set_experiment(EXPERIMENT_NAME)

# # @task(name='Starting Whylogs', retries=3)
# def starting_whylogs():
#     """
#     This function starts the Whylogs session.
#     Args:
#         None
#     Returns:
#         writer (whylogs.app.writers.WhyLabsWriter): Whylogs writer.
#         session (whylogs.app.Session): Whylogs session.
#     """
#     # k = keys_apis.Keys()
#     # k.obtain_whylogs_key()
#     # try:
#     #     os.environ["WHYLABS_API_KEY"] = "pnu_wiJT40R9TqGtTCplxQzdpM3VD5h9mb18AnQh"
#     # except Exception as e:
#     #     print("WHYLABS_API_KEY not found" + str(e))
#     #     os.environ["WHYLABS_API_KEY"] = os.environ["WHYLOGS_ACCOUNT_KEY"]
#     #     raise
#     os.environ["WHYLABS_DEFAULT_ORG_ID"] = "org-NLzasy"  # ORG-ID is case sensistive
#     os.environ["WHYLABS_API_KEY"] = "rU5GhR5P8w.8YO0h2NbOjJKDduWYnvWFRbzAODp6Uy5aWTVIOuc8BUHyobcMjxLE"
#     os.environ[
#         "WHYLABS_DEFAULT_DATASET_ID"
#     ] = "Sentiment-Analysis"  # The selected model project "My Model  (model-1)" is "model-1"
#     # Adding the WhyLabs Writer to utilize WhyLabs platform
#     writer = WhyLabsWriter("", formats=[])
#     session = Session(project="Sentiment-Analysis", pipeline="mlops-project-pipeline", writers=[writer])
#     return writer, session

# @task(name='Performance Metrics', retries=3)
# def performance_metrics(X_val, y_val, y_pred, session, logreg):
#     """
#     This function calculates the performance metrics and save it into Whylabs.
#     Args:
#         X_val (pandas.DataFrame): Validation features.
#         y_val (pandas.DataFrame): Validation labels.
#         y_pred (pandas.DataFrame): Predicted labels.
#         session (whylogs.app.Session): Whylogs session.
#         logreg (sklearn.linear_model.LogisticRegression): Logistic regression model.
#     Returns:
#         None
#     """
#     scores = [max(p) for p in logreg.predict_proba(X_val)]
#     with session.logger(tags={"datasetId": "model-1"}, dataset_timestamp=datetime.now()) as ylog:
#         ylog.log_metrics(
#             targets=list(y_val),
#             predictions=list(y_pred),
#             scores=scores,
#             model_type=ModelType.CLASSIFICATION,
#             target_field="Sentiment",
#             prediction_field="prediction",
#             score_field="Normalized Prediction Probability",
#         )
#     # closing the session
#     # session.close()

# @task(name="Create Pipeline", retries=3)
def create_pipeline(train_dicts, y_train):
    """
    Create a pipeline to train a model.
    Args:
        train_dicts : list of dicts
            The list of dictionaries to use for training.
        y_train : list of floats
            The list of target values to use for training.
    Returns:
        sklearn.pipeline.Pipeline:The pipeline to use for training.
    """
    pipeline = make_pipeline([('tfidf', TfidfVectorizer()), ('log_reg', LogisticRegression())])
    pipeline.fit(train_dicts, y_train)
    # Save the pipeline to a file
    with open("models/pipeline.bin", "wb") as f:
        pickle.dump(pipeline, f)


# @task(name="Extract_Data", retries=3)
def extract_train() -> pd.DataFrame:
    df = pd.read_csv("https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv")
    train_labels = df['label']
    return train_labels

def extract_data() -> pd.DataFrame:
    """
    Extract data from csv file and return dataframe with all the preprocessing operation on texts
    Returns:
        pd.DataFrame: dataframe with data
    """
    df = pd.read_csv("https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv")
    train_labels = df['label']
    #Remove "@user" from all the tweets as it is not providing any significant value add
    df['Tidy_Tweets'] = np.vectorize(remove_pattern)(df['tweet'], "@[\w]*")
    #Removing Punctuation, Numbers, and Special Characters
    df['Tidy_Tweets'] = df['Tidy_Tweets'].str.replace("[^a-zA-Z#]", " ")
    #Removing Short Words
    df['Tidy_Tweets'] = df['Tidy_Tweets'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
    #Tokenization
    tokenized_tweet = df['Tidy_Tweets'].apply(lambda x: x.split())
    tokenized_tweet = tokenized_tweet.apply(lambda x: [ps.stem(i) for i in x])
    for i in range(len(tokenized_tweet)):
        tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
    # Delete unnecessary columns
    df['Tidy_Tweets'] = tokenized_tweet

    # with session.logger(tags={"datasetId": "model-1"}, dataset_timestamp=datetime.now()) as ylog:
    #     ylog.log_dataframe(df)
    return df

def applying_tfidf():
    df = extract_data()
    tfidf=TfidfVectorizer(max_df=0.90, min_df=2,max_features=1000,stop_words='english')
    tfidf_matrix=tfidf.fit_transform(df['Tidy_Tweets'])
    df_tfidf = pd.DataFrame(tfidf_matrix.todense())
    train_tfidf_matrix = df_tfidf[:31962]
    # train_tfidf_matrix.todense()
    return train_tfidf_matrix


# @task(name="Transform_data", retries=3)
def transform_data(df, train_labels) -> pd.DataFrame:
    """
    Transform dataframe to get features and labels
    Args:
        df (pd.DataFrame): dataframe with data
    Returns:
        X_train (csr_matrix): features for training
        y_train (array): labels for training
        X_val (csr_matrix): features for validation
        y_val (array): labels for validation
    """
    ## Divide the data into train and test
    df_train_all, df_test, y_train_all, y_test = train_test_split(df, train_labels, test_size=0.25, random_state=0)
    ## Training model
    df_train, df_val, y_train, y_val  = train_test_split(df_train_all, y_train_all, test_size=0.25, random_state=0)
    return (
            df_train,
            df_val,
            y_train,
            y_val,
            df_test,
            y_test
    )

# @flow(name="Applying ML Model")
def applying_model():
    """
    Apply model to data
    Returns:
        None
    """
    # writer, session = starting_whylogs()
    # # df = extract_data(writer, session)
    df = applying_tfidf()
    print(df.shape)
    df_labels = extract_train()
    print(df_labels.shape)
    (
        df_train,
        df_val,
        y_train,
        y_val,
        df_test,
        y_test
    ) = transform_data(df, df_labels)
    with mlflow.start_run():
        # Create tags and log params
        mlflow.set_tag("model_type", "logistic_regression")
        mlflow.set_tag("developer", "Ajinkya Mishrikotkar")
        mlflow.log_param("train-data-path", "https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv")
        mlflow.log_param("val-data-path", "https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv")
        # Create Model
        logreg = LogisticRegression()
        logreg.fit(df_train, y_train)
        y_pred = logreg.predict(df_val)
        accuracy = accuracy_score(y_val, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_artifact(local_path="Models/model.pkl", artifact_path="models/logreg")
        # Model Register
        mlflow.sklearn.log_model(
            sk_model=logreg,
            artifact_path="models/logreg",
            registered_model_name="sk-learn-logreg-model",
        )
    # Capture permorfance metrics to show
    # performance_metrics(X_val, y_val, y_pred, session, logreg)

    return logreg



if __name__ == "__main__":
    """
    When you run this python script from the command line, it will run the flow
    Args:
        None
    Returns:
        None
    """
    logreg = applying_model()
    # Save model to pickle file
    with open("models/logreg.pkl", "wb") as f:
        pickle.dump(logreg, f)
    print("Model has been trained and saved")