import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn import svm
import sys

def read_data():
    # id	tweet	subtask_a	subtask_b	subtask_c
    training_data = pd.read_csv("../Data/olid-training-v1.0.tsv", sep='\t')
    test_data_text = pd.read_csv("../Data/testset-levela.tsv", sep='\t')
    test_data_labels = pd.read_csv("../Data/labels-levela.csv", header = None)
    test_data_labels.columns = ['id', 'subtask_a']
    return training_data, test_data_text, test_data_labels

def identity(x):
    return x

def main():
    training_data, test_data_text, test_data_labels = read_data()

    Xtrain = training_data['tweet'].tolist()
    Ytrain = training_data['subtask_a'].tolist()
    Xtest = test_data_text['tweet'].tolist()
    Ytest = test_data_labels['subtask_a'].tolist()

    vec = TfidfVectorizer(preprocessor = identity,
                          tokenizer = identity)

    clf = svm.SVC(kernel='linear', C=1.0)
    classifier = Pipeline( [('vec', vec), ('cls', clf)] )

    classifier.fit(Xtrain, Ytrain)

    Yguess = classifier.predict(Xtest)

    print(classification_report(Ytest, Yguess))

if __name__ == '__main__':
    main()
