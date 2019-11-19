import pandas as pd
from keras.models import Sequential
from sklearn.feature_extraction.text import CountVectorizer
from keras import layers
import sys
import numpy as np

def read_data():
    # id	tweet	subtask_a	subtask_b	subtask_c
    training_set = pd.read_csv("../Data/olid-training-v1.0.tsv", sep='\t')

    #training_set['subtask_c'] = training_set['subtask_c'].apply(lambda x: subtask_c_dict[x])
    training_data = training_set[(training_set.index < np.percentile(training_set.index, 80))]
    dev_data = training_set[(training_set.index > np.percentile(training_set.index, 80))]
    return training_data, dev_data

def main():
    training_data, dev_data = read_data()

    if len(sys.argv) > 1:
        if sys.argv[1].lower() == "--a":
            subtask_a_dict = {'NOT': 0, 'OFF' : 1}
            training_data['subtask_a'] = training_data['subtask_a'].apply(lambda x: subtask_a_dict[x])
            dev_data['subtask_a'] = dev_data['subtask_a'].apply(lambda x: subtask_a_dict[x])
            Xtrain = training_data['tweet'].tolist()
            Ytrain = training_data['subtask_a'].tolist()
            Xtest = dev_data['tweet'].tolist()
            Ytest = dev_data['subtask_a'].tolist()
        elif sys.argv[1].lower() == "--b":
            subtask_b_dict = {'UNT': 0, 'TIN': 1}
            training_data = training_data[training_data['subtask_a'] == 'OFF']
            dev_data = dev_data[dev_data['subtask_a'] == 'OFF']
            training_data['subtask_b'] = training_data['subtask_b'].apply(lambda x: subtask_b_dict[x])
            dev_data['subtask_b'] = dev_data['subtask_b'].apply(lambda x: subtask_b_dict[x])
            Xtrain = training_data['tweet'].tolist()
            Ytrain = training_data['subtask_b'].tolist()
            Xtest = dev_data['tweet'].tolist()
            Ytest = dev_data['subtask_b'].tolist()
        elif sys.argv[1].lower() == "--c":
            subtask_c_dict = {'OTH': 0, 'IND': 1, 'GRP': 2}
            training_data = training_data[training_data['subtask_b'] == 'TIN']
            dev_data = dev_data[dev_data['subtask_b'] == 'TIN']
            training_data['subtask_c'] = training_data['subtask_c'].apply(lambda x: subtask_c_dict[x])
            dev_data['subtask_c'] = dev_data['subtask_c'].apply(lambda x: subtask_c_dict[x])
            Xtrain = training_data['tweet'].tolist()
            Ytrain = training_data['subtask_c'].tolist()
            Xtest = dev_data['tweet'].tolist()
            Ytest = dev_data['subtask_c'].tolist()

        vectorizer = CountVectorizer()
        vectorizer.fit(Xtrain)

        Xtrain = vectorizer.transform(Xtrain)
        Xtest  = vectorizer.transform(Xtest)

        input_dim = Xtrain.shape[1]

        model = Sequential()
        model.add(layers.Dense(10, input_dim = input_dim, activation = 'relu'))
        model.add(layers.Dense(1, activation = 'sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

        history = model.fit(Xtrain, Ytrain,epochs = 5 ,verbose=True, validation_data=(Xtest, Ytest), batch_size=30)
        loss, accuracy = model.evaluate(Xtrain, Ytrain, verbose=False)
        print("Training Accuracy: {:.4f}".format(accuracy))
        loss, accuracy = model.evaluate(Xtest, Ytest, verbose=False)
        print("Testing Accuracy:  {:.4f}".format(accuracy))

    else:
        print("Enter a parameter")

if __name__ == '__main__':
    main()

# https://realpython.com/python-keras-text-classification/
