import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
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
import numpy as np
import matplotlib.pyplot as plt
from nltk import pos_tag
import hashtags
import emoji_to_words

def read_data():
    # id    tweet   subtask_a   subtask_b   subtask_c
    training_set = pd.read_csv("../Data/olid-training-v1.0.tsv", sep='\t')
    training_data = training_set[(training_set.index < np.percentile(training_set.index, 80))]
    dev_data = training_set[(training_set.index > np.percentile(training_set.index, 80))]
    #is_offensive =  training_data['subtask_a'] == 'OFF'
    #is_targeted = training_data['subtask_b'] == 'TIN'
    return training_data, dev_data

def identity(x):
    return x

def preprocess(tweet):
#    stemmer = SnowballStemmer('english')
#    lemmatizer = WordNetLemmatizer()
#    tweet = stemmer.stem(tweet)
#    tweet = lemmatizer.lemmatize(tweet)
    blacklistword = False
    whitelistword = False
    tknzr = TweetTokenizer()
    stop_words = stopwords.words('english')
    wordlist = tknzr.tokenize(tweet)
    wordlistwithoutstopwords = []
    for word in wordlist:
        if word not in stop_words or pos_tag([word])[0][1] == "PRP":
            wordlistwithoutstopwords.append(word)
        if use_blacklist:
            if word in blacklist:
                blacklistword = True
            else:
                if word in whitelist:
                    whitelistword = True
    if blacklistword:
        if use_blacklist:
            wordlistwithoutstopwords.append("BLACKLIST")   #Extra feature for blacklist
        else:
            pass
    else:
        if whitelistword:
            if use_blacklist:
                wordlistwithoutstopwords.append("WHITELIST") #Extra feature for whitelist
            else:
                pass
    toktweet = " ".join(wordlistwithoutstopwords)

#    for word in toktweet.split():
#        try:
#            toktweet=toktweet.replace(word,hashtags.do_splitting(word))
#        except:
#            print(toktweet, "do_splitting")
#        try:
#            toktweet=toktweet.replace(word,emoji_to_words.emoji_to_words(word))
#        except:
#            print(toktweet, "emoji")

    return toktweet

def tokenize(tweet):
    wordlist = tknzr.tokenize(tweet)
    return wordlist 

def blacklist_reader():
    file = open("../Data/offensive_words.txt", "r")
    f = file.read().strip()
    blacklist = f.split("\n")
    return blacklist

def whitelist_reader():
    file = open("../Data/not_offensive_words.txt", "r")
    f = file.read().strip()
    whitelist = f.split("\n")
    return whitelist

def print_n_most_informative_features(coefs, features, n):
    # Prints the n most informative features
    most_informative_feature_list = [(coefs[0][nr],feature) for nr, feature in enumerate(features)]
    sorted_most_informative_feature_list = sorted(most_informative_feature_list, key=lambda tup: abs(tup[0]), reverse=True)
    print("\nMOST INFORMATIVE FEATURES\n#\tvalue\tfeature")
    for nr, most_informative_feature in enumerate(sorted_most_informative_feature_list[:n]):
        print(str(nr+1) + ".","\t%.3f\t%s" % (most_informative_feature[0], most_informative_feature[1]))

def create_confusion_matrix(true, pred):
    # Build confusion matrix with matplotlib
    classes = sorted(list(set(pred)))
    # Build matrix
    cm = confusion_matrix(true, pred, labels = classes)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # Make plot
    plt.imshow(cm, interpolation = 'nearest', cmap=plt.cm.Blues)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.xlabel('Predicted label')
    plt.yticks(tick_marks, classes)
    plt.ylabel('True label')
    plt.show()

def main():
    training_data, dev_data = read_data()

    global use_blacklist
    use_blacklist = False

#    is_offensive =  training_data['subtask_a'] == 'OFF'

    if sys.argv[1].lower() == "--a":
        Xtrain = training_data['tweet'].tolist()
        Ytrain = training_data['subtask_a'].tolist()
        Xtest = dev_data['tweet'].tolist()
        Ytest = dev_data['subtask_a'].tolist()
        use_blacklist = True
        global blacklist
        global whitelist
        blacklist = blacklist_reader()
        whitelist = whitelist_reader()
    elif sys.argv[1].lower() == "--b":
        training_data = training_data[training_data['subtask_a'] == 'OFF']
        dev_data = dev_data[dev_data['subtask_a'] == 'OFF']
        Xtrain = training_data['tweet'].tolist()
        Ytrain = training_data['subtask_b'].tolist()
        Xtest = dev_data['tweet'].tolist()
        Ytest = dev_data['subtask_b'].tolist()
    elif sys.argv[1].lower() == "--c":
        training_data = training_data[training_data['subtask_b'] == 'TIN']
        dev_data = dev_data[dev_data['subtask_b'] == 'TIN']
        Xtrain = training_data['tweet'].tolist()
        Ytrain = training_data['subtask_c'].tolist()
        Xtest = dev_data['tweet'].tolist()
        Ytest = dev_data['subtask_c'].tolist()

    vec = TfidfVectorizer(tokenizer = tokenize,
                          preprocessor = preprocess,
                          analyzer = 'char_wb',
                          ngram_range = (1,9))

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
    print(confusion_matrix(Ytest, Yguess))
    create_confusion_matrix(Ytest, Yguess)
    pass

if __name__ == '__main__':
    main()
