import pandas as pd
from keras.models import Sequential
from sklearn.feature_extraction.text import CountVectorizer
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import layers
import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def read_data():
    # id	tweet	subtask_a	subtask_b	subtask_c
    training_set = pd.read_csv("../Data/olid-training-v1.0.tsv", sep='\t')

    #training_set['subtask_c'] = training_set['subtask_c'].apply(lambda x: subtask_c_dict[x])
    training_data = training_set[(training_set.index < np.percentile(training_set.index, 80))]
    dev_data = training_set[(training_set.index > np.percentile(training_set.index, 80))]
    return training_data, dev_data

def create_model(num_filters, kernel_size, vocab_size, embedding_dim, maxlen):
    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
    model.add(layers.Conv1D(num_filters, kernel_size, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

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

        # Main settings
        epochs = 5
        embedding_dim = 50
        maxlen = 100
        output_file = 'data/output.txt'

        # Tokenize words
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(Xtrain)
        Xtrain = tokenizer.texts_to_sequences(Xtrain)
        Xtest = tokenizer.texts_to_sequences(Xtest)

        # Adding 1 because of reserved 0 index
        vocab_size = len(tokenizer.word_index) + 1

        # Pad sequences with zeros
        Xtrain = pad_sequences(Xtrain, padding='post', maxlen=maxlen)
        Xtest = pad_sequences(Xtest, padding='post', maxlen=maxlen)

        # Parameter grid for grid search
        param_grid = dict(num_filters=[32, 64, 128],
                          kernel_size=[3, 5, 7],
                          vocab_size=[vocab_size],
                          embedding_dim=[embedding_dim],
                          maxlen=[maxlen])

        model = KerasClassifier(build_fn=create_model,
                                epochs=epochs, batch_size=10,
                                verbose=False)

        grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid,
                                  cv=2, verbose=1, n_iter=5)

        grid_result = grid.fit(Xtrain, Ytrain)

        # Evaluate testing set
        test_accuracy = grid.score(Xtest, Ytest)
        print(test_accuracy)
        print(grid_result.best_score_)
        print(grid_result.best_params_)

    else:
        print("Enter a parameter")

if __name__ == '__main__':
    main()

# https://realpython.com/python-keras-text-classification/
