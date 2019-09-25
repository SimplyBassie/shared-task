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

def read_data():
    # id	tweet	subtask_a	subtask_b	subtask_c
    with open('../Data/olid-training-v1.0.tsv') as tsvfile:
      file = csv.reader(tsvfile, delimiter='\t')
      trainingdata = pd.DataFrame(file, columns=['id', 'tweet', 'subtask_a', 'subtask_b', 'subtask_c'])

    return trainingdata

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
                            ('cls', KNeighborsClassifier(n_neighbors=63))] )

    classifier.fit(Xtrain, Ytrain)

    Yguess = classifier.predict(Xtest)

    print(classification_report(Ytest, Yguess))

if __name__ == '__main__':
    main()
