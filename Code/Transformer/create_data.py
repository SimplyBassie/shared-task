import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook
import numpy as np
import sys

prefix = 'data/'

training_set = pd.read_csv(prefix + 'olid-training-v1.0.tsv', sep='\t')
train_df = training_set[(training_set.index < np.percentile(training_set.index, 80))]
test_df = training_set[(training_set.index > np.percentile(training_set.index, 80))]

if len(sys.argv) > 1:
    if sys.argv[1].lower() == "--a":
        subtask_a_dict = {'NOT': 0, 'OFF' : 1}
        train_df['subtask_a'] = train_df['subtask_a'].apply(lambda x: subtask_a_dict[x])
        test_df['subtask_a'] = test_df['subtask_a'].apply(lambda x: subtask_a_dict[x])

        train_df = pd.DataFrame({
            'id':range(len(train_df)),
            'label':train_df['subtask_a'],
            'alpha':['a']*train_df.shape[0],
            'text': train_df['tweet'].replace(r'\n', ' ', regex=True)
        })

        dev_df = pd.DataFrame({
            'id':range(len(test_df)),
            'label':test_df['subtask_a'],
            'alpha':['a']*test_df.shape[0],
            'text': test_df['tweet'].replace(r'\n', ' ', regex=True)
        })

    elif sys.argv[1].lower() == "--b":
        subtask_b_dict = {'UNT': 0, 'TIN': 1}
        train_df = train_df[train_df['subtask_a'] == 'OFF']
        test_df = test_df[test_df['subtask_a'] == 'OFF']
        train_df['subtask_b'] = train_df['subtask_b'].apply(lambda x: subtask_b_dict[x])
        test_df['subtask_b'] = test_df['subtask_b'].apply(lambda x: subtask_b_dict[x])

        train_df = pd.DataFrame({
            'id':range(len(train_df)),
            'label':train_df['subtask_b'],
            'alpha':['a']*train_df.shape[0],
            'text': train_df['tweet'].replace(r'\n', ' ', regex=True)
        })

        dev_df = pd.DataFrame({
            'id':range(len(test_df)),
            'label':test_df['subtask_b'],
            'alpha':['a']*test_df.shape[0],
            'text': test_df['tweet'].replace(r'\n', ' ', regex=True)
        })

    elif sys.argv[1].lower() == "--c":
        subtask_c_dict = {'OTH': 0, 'IND': 1, 'GRP': 2}
        train_df = train_df[train_df['subtask_b'] == 'TIN']
        test_df = test_df[test_df['subtask_b'] == 'TIN']
        train_df['subtask_c'] = train_df['subtask_c'].apply(lambda x: subtask_c_dict[x])
        test_df['subtask_c'] = test_df['subtask_c'].apply(lambda x: subtask_c_dict[x])

        train_df = pd.DataFrame({
            'id':range(len(train_df)),
            'label':train_df['subtask_c'],
            'alpha':['a']*train_df.shape[0],
            'text': train_df['tweet'].replace(r'\n', ' ', regex=True)
        })

        dev_df = pd.DataFrame({
            'id':range(len(test_df)),
            'label':test_df['subtask_c'],
            'alpha':['a']*test_df.shape[0],
            'text': test_df['tweet'].replace(r'\n', ' ', regex=True)
        })

train_df.to_csv('data/train.tsv', sep='\t', index=False, header=False)
dev_df.to_csv('data/dev.tsv', sep='\t', index=False, header=False)
