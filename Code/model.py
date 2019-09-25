import csv
import pandas as pd

def read_data():
    # id	tweet	subtask_a	subtask_b	subtask_c
    with open('../Data/olid-training-v1.0.tsv') as tsvfile:
      file = csv.reader(tsvfile, delimiter='\t')
      trainingdata = pd.DataFrame(file, columns=['id', 'tweet', 'subtask_a', 'subtask_b', 'subtask_c'])

    return trainingdata

def main():
    dataframe = read_data()


if __name__ == '__main__':
    main()
