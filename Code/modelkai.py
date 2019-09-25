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
    stopwordlist = stopwords.words('english')
    stemmer = SnowballStemmer('english')
    lemmatizer = WordNetLemmatizer()
    print("aapje")

    documents = []
    labels = []
    with open("../Data/olid-training-v1.0.tsv", encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split('\t')

            for word in tokens[1].split():
                if word in stopwordlist:
                    tokens[1] = tokens[1].replace(" "+word+" ", " ")
                else:
                    tokens[1] = tokens[1].replace(" "+word+" ", " "+stemmer.stem(word)+" ")

            documents.append(tokens[1])
            labels.append(tokens[2] )

    print(documents)
    print(labels)

    return documents, labels

def identity(x):
    return x

def main():
    X, Y = read_data()
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
