import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn import svm
import sys
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer

def read_data():
    # id	tweet	subtask_a	subtask_b	subtask_c
    training_data = pd.read_csv("../Data/olid-training-v1.0.tsv", sep='\t')
    test_data_text = pd.read_csv("../Data/testset-levela.tsv", sep='\t')
    test_data_labels = pd.read_csv("../Data/labels-levela.csv", header = None)
    test_data_labels.columns = ['id', 'subtask_a']
    return training_data, test_data_text, test_data_labels

def identity(x):
    return x

def preprocess(tweet):
    stemmer = SnowballStemmer('english')
    lemmatizer = WordNetLemmatizer()
    pptweetlist = []
    tweet = stemmer.stem(tweet)
    tweet = lemmatizer.lemmatize(tweet)
    return tweet

def tokenize(tweet):
    tknzr = TweetTokenizer()
    stop_words = stopwords.words('english')
    wordlist = tknzr.tokenize(tweet)
    wordlistwithoutstopwords = []
    for word in wordlist:
        if word not in stop_words:
            wordlistwithoutstopwords.append(word)
    toktweet = " ".join(wordlistwithoutstopwords)
    return toktweet


def main():
    training_data, test_data_text, test_data_labels = read_data()

    Xtrain = training_data['tweet'].tolist()
    Ytrain = training_data['subtask_a'].tolist()
    Xtest = test_data_text['tweet'].tolist()
    Ytest = test_data_labels['subtask_a'].tolist()

    vec = TfidfVectorizer(preprocessor = preprocess,
                          tokenizer = tokenize,
                          ngram_range = (1,5))

    clf = svm.LinearSVC(C=1.0)
    classifier = Pipeline( [('vec', vec), ('cls', clf)] )

    classifier.fit(Xtrain, Ytrain)

    Yguess = classifier.predict(Xtest)

    print(classification_report(Ytest, Yguess))

if __name__ == '__main__':
    main()
