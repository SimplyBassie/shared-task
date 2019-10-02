import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
import sys
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

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
    return wordlistwithoutstopwords

def blacklist_reader():
    file = open("../Data/offensive_words.txt", "r")
    f = file.read().strip()
    blacklist = f.split("\n")
    return blacklist

def print_n_most_informative_features(coefs, features, n):
    # Prints the n most informative features
    most_informative_feature_list = [(coefs[0][nr],feature) for nr, feature in enumerate(features)]
    sorted_most_informative_feature_list = sorted(most_informative_feature_list, key=lambda tup: abs(tup[0]), reverse=True)
    print("\nMOST INFORMATIVE FEATURES\n#\tvalue\tfeature")
    for nr, most_informative_feature in enumerate(sorted_most_informative_feature_list[:n]):
        print(str(nr+1) + ".","\t%.3f\t%s" % (most_informative_feature[0], most_informative_feature[1]))


def main():
    training_data, test_data_text, test_data_labels = read_data()
    blacklist = blacklist_reader()

    Xtrain = training_data['tweet'].tolist()
    Ytrain = training_data['subtask_a'].tolist()
    Xtest = test_data_text['tweet'].tolist()
    Ytest = test_data_labels['subtask_a'].tolist()

    vec = TfidfVectorizer(tokenizer = tokenize,
                          preprocessor = preprocess,
                          ngram_range = (1,5))

    #clf1 = DecisionTreeClassifier(max_depth=20)
    #clf2 = KNeighborsClassifier(n_neighbors=9)
    clf3 = LinearSVC(C=1)
    #ens = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3)], voting='hard', weights=[1, 1, 1])

    classifier = Pipeline( [('vec', vec), ('cls', clf3)] )

    classifier.fit(Xtrain, Ytrain)

    coefs = classifier.named_steps['cls'].coef_
    features = classifier.named_steps['vec'].get_feature_names()
    print_n_most_informative_features(coefs, features, 10)

    Yguess = classifier.predict(Xtest)

    print(classification_report(Ytest, Yguess))

if __name__ == '__main__':
    main()
