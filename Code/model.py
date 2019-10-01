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
    # id	tweet	subtask_a	subtask_b	subtask_c
    with open('../Data/olid-training-v1.0.tsv') as tsvfile:
      file = csv.reader(tsvfile, delimiter='\t')
      trainingdata = pd.DataFrame(file)

    return trainingdata

def identity(x):
    return x

def main():
    dataframe = read_data()
    print(dataframe.head())
    #Xtrain = dataframe_train['tweet'].tolist()
    #Ytrain = dataframe_train['subtask_a'].tolist()
    #Xtest =

    vec = TfidfVectorizer(preprocessor = identity,
                          tokenizer = identity)

    #clf = svm.SVC(kernel='linear', C=1.0)
    #classifier = Pipeline( [('vec', vec),
                            #('cls', MultinomialNB())] )
                            #('cls', DecisionTreeClassifier(max_depth=60, min_impurity_decrease=0.001))] )
                            #('cls', KNeighborsClassifier(n_neighbors=63))] )
                            ('cls', clf)] )

    #classifier.fit(Xtrain, Ytrain)

    #Yguess = classifier.predict(Xtest)

    #print(classification_report(Ytest, Yguess))

if __name__ == '__main__':
    main()
