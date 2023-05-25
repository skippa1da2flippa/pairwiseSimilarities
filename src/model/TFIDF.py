import os
from time import time
from typing import Union
from nltk.corpus import stopwords
from numpy import ndarray, array, append, argsort, sqrt, matrix
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from src.model.dataManager import DataManager
from nltk import word_tokenize, WordNetLemmatizer
import re

"""
    In this function you should take randomly a subset of the documents collection, by using 
    a uniform probability distribution
"""


def takeASubSet():
    # TODO
    pass


def normalize(tfIdfMatrix: csr_matrix):
    normArray: matrix = sqrt(tfIdfMatrix.multiply(tfIdfMatrix).sum(1))
    for idx in range(0, tfIdfMatrix.shape[0]):
        tfIdfMatrix[idx, :] = tfIdfMatrix[idx, :] / normArray[idx, 0]


def preprocessing(data: ndarray[dict[str, str]]) -> tuple[ndarray[str], dict[str, int]]:
    stop_words = set(stopwords.words("english"))
    lem = WordNetLemmatizer()
    corpus: ndarray[str] = array([])
    docsLen: dict[str, int] = {}
    for doc in data:
        actualDoc: str = re.sub(r'[^\w\s]', '', doc["title"] + ". " + doc["text"])
        tokenizedDoc: list[str] = word_tokenize(actualDoc)
        finalDoc: list[str] = [lem.lemmatize(word) for word in tokenizedDoc if word.lower() not in stop_words]
        docsLen[doc["_id"]] = len(finalDoc)
        corpus = append(corpus, " ".join(finalDoc))

    return corpus, docsLen


"""
    I sort by idf because if we are computing cosine similarity and our similarity score is still
    low we can stop. This because the deeper we go in each document the less score the related word
    has (low df). Furthermore I sort the matrix rows by documents length because if two documents have a 
    big ass difference in length, I don't even check them with the cosine similarities
"""


def tfIdfMatrixGetter(docCollection: ndarray[dict[str, str]]) -> tuple[csr_matrix, dict[str, int]]:
    corpus, docsLen = preprocessing(docCollection)

    # instantiate TFIDF class
    tfIdf = TfidfVectorizer()

    # fit data to compute IDF for each word in the vocabulary
    tfIdf.fit(corpus)

    # sort IDF (getting the indexes)
    sortedIds: ndarray[int] = argsort(tfIdf.idf_)

    # generates TFIDF matrix
    tfIdfMatrix: csr_matrix = tfIdf.transform(corpus)

    # sort the matrix columns by idf
    tfIdfMatrix = tfIdfMatrix[:, sortedIds]

    # normalize each row with its euclidean norm
    normalize(tfIdfMatrix)

    return tfIdfMatrix, docsLen


def computeLenRatio(fstLen: int, sdnLen: int) -> float:
    return fstLen / sdnLen if sdnLen > fstLen else sdnLen / fstLen


def pairWiseSimilarity(tfIdfMatrix: csr_matrix, docsLen: dict[str, int], threshold: float, hyperLen: float = 0.5) -> \
        dict[str, Union[list[tuple[float, tuple[str, str]]], float]]:

    start = time()

    pairs: list[tuple[float, tuple[str, str]]] = []
    lens: list[int] = list(docsLen.values())
    docIds: list[str] = list(docsLen.keys())

    for x in range(0, tfIdfMatrix.shape[0] - 1):
        for y in range(x + 1, tfIdfMatrix.shape[0]):
            if computeLenRatio(lens[x], lens[y]) >= hyperLen:
                fstArray: ndarray[float] = tfIdfMatrix[x, :].toarray()[0, :]
                sdnArray: ndarray[float] = tfIdfMatrix[y, :].toarray()[0, :]
                cosineSimilarity: float = fstArray @ sdnArray
                if cosineSimilarity >= threshold:
                    pairs.append((cosineSimilarity, (docIds[x], docIds[y])))

    end = time()

    return {
        "result": pairs,
        "time": end - start
    }


def startProcess():
    print(os.listdir("."))  # return all the elements in the cwd
    basePath: str = "src/model/data/"
    collector: DataManager = DataManager(
        basePath + "corpus.jsonl",
    )

    tfIdfMatrix, docsLen = tfIdfMatrixGetter(collector.getDocuments()[:5])

    print(pairWiseSimilarity(tfIdfMatrix, docsLen, 0.1))
