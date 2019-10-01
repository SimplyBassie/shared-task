import csv
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from sklearn import svm


def read_data():
    # id    tweet   subtask_a   subtask_b   subtask_c
    with open('../Data/olid-training-v1.0.tsv') as tsvfile:
      file = csv.reader(tsvfile, delimiter='\t')
      trainingdata = pd.DataFrame(file, columns=['id', 'tweet', 'subtask_a', 'subtask_b', 'subtask_c'])

    return trainingdata


def print_n_most_informative_features(coefs, features, n):
    # Prints the n most informative features
    most_informative_feature_list = [(coefs[0][nr],feature) for nr, feature in enumerate(features)]
    sorted_most_informative_feature_list = sorted(most_informative_feature_list, key=lambda tup: abs(tup[0]), reverse=True)
    print("\nMOST INFORMATIVE FEATURES\n#\tvalue\tfeature")
    for nr, most_informative_feature in enumerate(sorted_most_informative_feature_list[:n]):
        print(str(nr+1) + ".","\t%.3f\t%s" % (most_informative_feature[0], most_informative_feature[1]))


def identity(x):
    return x

def main():
    dataframe = read_data()
    X = dataframe['tweet'].tolist()
    Y = dataframe['subtask_a'].tolist()

    split_point = int(0.75*len(X))
    Xtrain = X[:split_point]
    Ytrain = Y[:split_point]
    Xtest = X[split_point:]
    Ytest = Y[split_point:]

    vec = TfidfVectorizer(preprocessor = identity,
                          tokenizer = identity)

    classifier = Pipeline( [('vec', vec),
                            #('cls', MultinomialNB())] )
                            #('cls', DecisionTreeClassifier(max_depth=60, min_impurity_decrease=0.001))] )
                            ('cls', svm.LinearSVC(C=10))] )

    classifier.fit(Xtrain, Ytrain)

    coefs = classifier.named_steps['cls'].coef_
    features = classifier.named_steps['vec'].get_feature_names()
    print_n_most_informative_features(coefs, features, 10)
    print()

    Yguess = classifier.predict(Xtest)

    print(classification_report(Ytest, Yguess))

if __name__ == '__main__':
    main()
