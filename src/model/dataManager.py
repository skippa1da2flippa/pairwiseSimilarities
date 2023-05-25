import json
from numpy import ndarray, array, append
from pandas import DataFrame, read_csv

"""
    Class implementing methods for fetching the dataset from a file 
"""


class DataManager(object):
    def __init__(self, documentsPath: str):
        self._documents: ndarray[dict[str, str]] = array([])
        self._readJson(documentsPath)

    """
        This method read from a file expecting it to contain a list of json
        and store it in a ndarray[dict[str, str]] 
    """

    def _readJson(self, path: str):
        with open(path, "r") as file:
            data: str = file.read()
            objects: list[str] = data.strip().split('\n')
            for obj in objects:
                try:
                    json_data = json.loads(obj)
                    self._documents = append(self._documents, json_data)
                except json.JSONDecodeError:
                    print("We got some issue with the json encoding")

    def getDocuments(self) -> ndarray[dict[str, str]]:
        return self._documents
