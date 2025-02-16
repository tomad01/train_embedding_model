import sqlite3, os, json, logging
from typing import List, Dict, Optional
import pandas as pd
from abc import abstractmethod
from azure.cosmos import CosmosClient, exceptions
logger = logging.getLogger("azure.core.pipeline.policies.http_logging_policy")
logger.setLevel(logging.WARNING)

sqlite_fields = {
    "machine": "TEXT",
    "experiment_timestamp": "TEXT",
    "experiment_name": "TEXT",
    "test_size": "INTEGER",
    "dataset": "TEXT",
    "tags": "TEXT",
    "model_params": "TEXT",
    "metrics": "TEXT",
    "classification_report": "TEXT",
    "confusion_matrix": "TEXT",
    "target_names": "TEXT",
    "feature_importance": "TEXT",
    "experiment_duration_hours": "FLOAT",
    "model_location": "TEXT",
    "ram_footprint_gb": "FLOAT",
}

class DataInterface:

    @abstractmethod
    def save_data(self, record:dict)->None:
        pass

    @abstractmethod
    def get_data(self, fields:List[str],filter_include:str=None,filter_exclude:str=None)->pd.DataFrame:
        pass

    @abstractmethod
    def update_experiment(self, field:str, value:str, where_column:str, where_value:str)->None:
        pass

    @abstractmethod
    def delete_experiment(self, timestamp:str)->None:
        pass

    @abstractmethod
    def get_item(self, value:str)->pd.DataFrame:
        pass

    def get_data_for_dashboard(self,timestamp=None,filter_include=None,filter_exclude=None):
        if timestamp:
            idf = self.get_item(timestamp)
        else:
            items = ['experiment_timestamp','classification_report','experiment_name','tags','metrics','ram_footprint_gb','experiment_duration_hours']
            idf = self.get_data(items,filter_include,filter_exclude)
        # dff = df[df.apply(lambda row: row.str.contains(include, regex=False).any(), axis=1)]
        # if len(exclude)>0:
        #     dff = dff[~dff.apply(lambda row: row.str.contains(exclude, regex=False).any(), axis=1)]
        # print(len(dff))
        idf["experiment_timestamp"] = pd.to_datetime(idf["experiment_timestamp"])
        idf['accuracy'] = idf['metrics'].apply(lambda x: json.loads(x)['accuracy'])
        idf['classes_nr'] = idf['classification_report'].apply(lambda x: len(json.loads(x)['precision'].keys())-3)
        if not timestamp:
            del idf['classification_report']
        for col in ['experiment_duration_hours','ram_footprint_gb','accuracy']:
            idf[col] = idf[col].round(2)
        del idf['metrics']
        return idf.sort_values('accuracy', ascending=False)
    
class CosmosDBWrapper(DataInterface):
    def __init__(self, endpoint: str = os.environ.get("COSMOS_ENDPOINT"),
                 key: str = os.environ.get("COSMOS_KEY"),
                 database_name: str = os.environ.get("COSMOS_DATABASE"),
                 container_name: str = "experiments"):
        self.client = CosmosClient(endpoint, {'masterKey': key})
        self.database = self.client.get_database_client(database_name)
        self.container = self.database.get_container_client(container_name)

    def save_data(self, record: Dict) -> None:
        """ Inserts or updates a record in the Cosmos DB container."""
        try:
            record['id'] = record['experiment_timestamp']
            self.container.create_item(record)
            self.container.upsert_item(record)
        except exceptions.CosmosHttpResponseError as e:
            print(f"Error saving data: {e}")

    def get_data(self, fields: List[str], filter_include: Optional[str] = None, filter_exclude: Optional[str] = None) -> pd.DataFrame:
        """ Retrieves data from the container, optionally filtering based on a substring match."""
        if fields[0]=="*":
            select_fields = "*"
        else:
            select_fields = ", ".join([f"c.{field}" for field in fields])

        query = f"SELECT {select_fields} FROM experiments c"
        
        if filter_include:
            query += f" WHERE CONTAINS(c.experiment_name, '{filter_include}')"
        elif filter_exclude:
            query += f" WHERE NOT CONTAINS(c.experiment_name, '{filter_exclude}')"
        
        items = list(self.container.query_items(query, enable_cross_partition_query=True))
        return pd.DataFrame(items)

    def update_experiment(self, field: str, value: str, where_column: str, where_value: str) -> None:
        """ Updates a specific field in a document where a condition is met."""
        query = f"SELECT * FROM experiments c WHERE c.{where_column} = @where_value"
        parameters = [{"name": "@where_value", "value": where_value}]
        items = list(self.container.query_items(query, parameters=parameters, enable_cross_partition_query=True))
        
        for item in items:
            item[field] = value
            self.container.upsert_item(item)

    def delete_experiment(self, timestamp: str) -> None:
        """ Deletes a document based on a timestamp."""
        query = "SELECT * FROM experiments c WHERE c.experiment_timestamp = @timestamp"
        parameters = [{"name": "@timestamp", "value": timestamp}]
        items = list(self.container.query_items(query, parameters=parameters, enable_cross_partition_query=True))
        
        for item in items:
            self.container.delete_item(item, partition_key=item['id'])

    def get_item(self, value: str) -> pd.DataFrame:
        """ Retrieves a single item based on an identifier."""
        query = "SELECT * FROM experiments c WHERE c.experiment_timestamp = @value"
        parameters = [{"name": "@value", "value": value}]
        items = list(self.container.query_items(query, parameters=parameters, enable_cross_partition_query=True))
        return pd.DataFrame(items)
    
class SqliteDataInterface(DataInterface):
    def __init__(self) -> None:
        self.conn_str = os.environ.get("MLOPS_CONN_STR", "/Users/DToma/work/classification_model/registry/benchmark.db")
        self.__create_schema_if_not_exists()

    def __execute_query(self, query, values=None):
        with sqlite3.connect(self.conn_str) as conn:
            cursor = conn.cursor()  # Create a cursor object
            if values:
                cursor.execute(query, values)
            else:
                cursor.execute(query)
            conn.commit()
            cursor.close()

    def save_data(self, record:dict):
        arg1 = ", ".join(record.keys())
        arg2 = ", ".join(["?" for _ in record.keys()])
        insert_query = f"INSERT INTO model_performance ({arg1}) VALUES ({arg2})"
        values = [record[k] for k in record.keys()]
        self.__execute_query(insert_query, values)

    def get_item(self, value:str):
        with sqlite3.connect(self.conn_str) as conn:
            idf = pd.read_sql(f"SELECT * FROM model_performance WHERE experiment_timestamp = '{value}'", conn)
        return idf

    def get_data(self, fields:List[str],filter_include:str=None,filter_exclude:str=None)->pd.DataFrame:
        fields = ", ".join(fields)
        with sqlite3.connect(self.conn_str) as conn:
            if filter_include is not None and filter_exclude is None:
                df = pd.read_sql_query(f"SELECT {fields} FROM model_performance WHERE experiment_name like '%{filter_include}%'", conn)
            elif filter_exclude is not None and filter_include is None:
                df = pd.read_sql_query(f"SELECT {fields} FROM model_performance WHERE experiment_name not like '%{filter_exclude}%'", conn)
            elif filter_exclude is not None and filter_include is not None:
                df = pd.read_sql_query(f"SELECT {fields} FROM model_performance WHERE experiment_name like '%{filter_include}%' and experiment_name not like '%{filter_exclude}%'", conn)
            else:
                df = pd.read_sql_query(f"SELECT {fields} FROM model_performance", conn)

        return df

    def __create_schema_if_not_exists(self):
        table_fields = ", ".join([f"{k} {v}" for k, v in sqlite_fields.items()])
        create_table_query = f"CREATE TABLE IF NOT EXISTS model_performance ({table_fields})"
        self.__execute_query(create_table_query)
    
    def update_experiment(self, field, value, where_column, where_value):
        update_query = f"UPDATE model_performance SET {field} = ? WHERE {where_column} = ?"
        self.__execute_query(update_query, (value, where_value))

    def delete_experiment(self, timestamp):
        self.__execute_query(f"DELETE FROM model_performance WHERE experiment_timestamp = '{timestamp}'")

