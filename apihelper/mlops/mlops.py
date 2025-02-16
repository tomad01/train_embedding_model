import time
import socket
import json
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report,
    f1_score,
    confusion_matrix,
    accuracy_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
import psutil
from dotenv import load_dotenv
load_dotenv("../../.env",override=True)
from typing import Literal
from .mlops_data_interface import SqliteDataInterface, CosmosDBWrapper



try:
    import matplotlib.pyplot as plt
except:
    pass


class Record:

    """
    this class is used to record the model performance metrics
    it stores data in sqlite database
    """

    def __init__(self,data_iterface:Literal['sqlite','cosmos']='sqlite') -> None:
        if data_iterface == 'sqlite':
            self.data_interface = SqliteDataInterface()
        elif data_iterface == 'cosmos':
            self.data_interface = CosmosDBWrapper()
        else:
            raise ValueError(f"Data interface {data_iterface} not supported")
        self.record = {}
        self.tags = []

    def get_ram_footprint(self):
        # Get the current process
        current_process = psutil.Process()

        # Get the memory info of the current process
        memory_info = current_process.memory_info()
        ram_footprint = memory_info.rss / (1024**3)
        return ram_footprint

    def model_location(self, location):
        self.record["model_location"] = location

    def dataset(self, dataset):
        self.record["dataset"] = dataset

    def add_tag(self, tag):
        self.tags.append(tag)
        self.record["tags"] = ','.join(set(self.tags))

    def model_params(self, model_params:dict):
        self.record["model_params"] = json.dumps(model_params)

    def begin_experiment(self, experiment_name=""):
        self.start = time.time()
        self.record["machine"] = socket.gethostname()
        self.record["experiment_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.record["experiment_name"] = experiment_name

    def update_experiment(self, field, value, where_column, where_value):
        self.data_interface.update_experiment(field, value, where_column, where_value)

    def end_experiment(self):
        self.record["ram_footprint_gb"] = self.get_ram_footprint()
        self.record["experiment_duration_hours"] = (time.time() - self.start) / 3600

        self.data_interface.save_data(self.record)
        # self.record = {}
        self.tags = []

    def get_scores(self, like=None):
        df = self.data_interface.get_data(['experiment_name','tags','metrics','ram_footprint_gb'])
        # df['experiment_timestamp'] = pd.to_datetime(df['experiment_timestamp'])
        if like is not None:
            df = df[df["experiment_name"].str.contains(like)]

        return df  # .sort_values('experiment_timestamp',ascending=False)

    def get_records(self, like=None):
        df = self.data_interface.get_data(['*'],filter_include=like)
        if len(df)>0:
            df["experiment_timestamp"] = pd.to_datetime(df["experiment_timestamp"])
            return df.sort_values("experiment_timestamp", ascending=False)
        else:
            return df
    

    def log_metrics(
        self,
        y_pred,
        y_test,
        target_names=None,
        model = None,
        metrics=["classification_report", "confusion_matrix", "f1", "accuracy"],

    ):
        metrics_set = {}
        if target_names:
            self.record["target_names"] = json.dumps(target_names)
        if "classification_report" in metrics:
            report = classification_report(y_test, y_pred, output_dict=True, target_names=target_names)
            report = pd.DataFrame(report).T.sort_values("f1-score", ascending=False)
            self.record["classification_report"] = report.to_json()
        if "confusion_matrix" in metrics:
            self.record["confusion_matrix"] = json.dumps(confusion_matrix(y_test, y_pred).tolist())
        if "f1" in metrics:
            metrics_set["f1_score"] = f1_score(y_test, y_pred, average="macro")
            print(f"f1_score: {metrics_set['f1_score']}")
        if "accuracy" in metrics:
            metrics_set["accuracy"] = accuracy_score(y_test, y_pred)

        if "r2" in metrics:
            metrics_set["r2_score"] = r2_score(y_test, y_pred)
        if "mse" in metrics:
            metrics_set["mean_squared_error"] = mean_squared_error(y_test, y_pred)
        if "mae" in metrics:
            metrics_set["mean_absolute_error"] = mean_absolute_error(y_test, y_pred)
        if "mape" in metrics:
            metrics_set["mean_absolute_percentage_error"] = mean_absolute_percentage_error(y_test, y_pred)
        self.record["metrics"] = json.dumps(metrics_set)
        self.record["test_size"] = len(y_test)
        try:
            if hasattr(model, "get_params"):
                self.record["model_params"] = json.dumps(model.get_params())
            elif hasattr(model, "parameters"):
                self.record["model_params"] = json.dumps(model.parameters)
            elif hasattr(model, "params"):
                self.record["model_params"] = json.dumps(model.params)
            else:
                self.record["model_params"] = json.dumps(str(model.__dict__))
        except Exception as e:
            print(f"Error logging model params: {e}")

    def log_feature_importances(self, feature_importances, feature_names):
        log = pd.DataFrame({"Feature": feature_names, "Importance": feature_importances})

        # Sort the DataFrame by importance in descending order
        top_features = log.sort_values("Importance", ascending=False).head(16)
        top_features = top_features.sort_values("Importance", ascending=True)
        self.record["feature_importance"] = top_features.to_json()

    def delete_experiment(self, timestamp):
        self.data_interface.delete_experiment(timestamp)

    def plot_confusion_matrix(self, experiment_name):
        df = self.get_records(experiment_name)
        cm = pd.read_json(df["confusion_matrix"].values[0])
        target_names = json.loads(df["target_names"].values[0])
        # Set the figure size
        plt.figure(figsize=(10, 6))
        # use target names
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.show()

    def plot_feature_importance(self, experiment_name):
        df = self.get_records(experiment_name)
        fi = pd.read_json(df["feature_importance"].values[0])
        # Get the top 10 most important features
        # Set the figure size
        plt.figure(figsize=(10, 6))

        # Create a bar plot
        plt.barh(fi["Feature"], fi["Importance"])

        # Set labels and title
        plt.ylabel("Feature")
        plt.xlabel("Importance")
        plt.title("Feature Importances")

        # Show the plot
        plt.show()


if __name__ == "__main__":

    record = Record("cosmos")
    items = record.get_records(like='cioco')
    print(items)
    # from sklearn.ensemble import RandomForestClassifier

    # X = np.random.rand(100,8)
    # y = np.random.randint(0,2,100)
    # model = RandomForestClassifier()
    # model.fit(X,y)
    # y_pred = model.predict(X)
    # record.begin_experiment('test')
    # record.log_metrics(y_pred,y,['a','b'])
    # record.log_feature_importances(model.feature_importances_, ['a','b','c','d','e','f','g','h'])
    # record.end_experiment()
    # print(record.get_scores())
    # record.update_experiment(
    #     "experiment_name",
    #     "Escalation Prediction using bag encoder",
    #     "experiment_timestamp",
    #     "2024-08-08 22:37:48",
    # )
    # record.update_experiment(
    #     "experiment_name",
    #     "Escalation Prediction using bag encoder",
    #     "experiment_timestamp",
    #     "2024-08-08 22:34:27",
    # )
    # df = record.get_scores("Category")
    # with open("records.html", "w") as fo:
    #     fo.write(df.to_html())
