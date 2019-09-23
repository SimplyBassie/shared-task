import csv
import pandas as pd

from nltk.tokenize import word_tokenize
import nltk

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.model_selection import cross_validate

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn import metrics

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

def read_data():
    # id	tweet	subtask_a	subtask_b	subtask_c
    with open('../Data/olid-training-v1.0.tsv') as tsvfile:
      file = csv.reader(tsvfile, delimiter='\t')
      trainingdata = pd.DataFrame(file, columns=['id', 'tweet', 'subtask_a', 'subtask_b', 'subtask_c'])

    return trainingdata

def tokenize(text):
    return word_tokenize(text)

def main():
    dataframe = read_data()


if __name__ == '__main__':
    main()
