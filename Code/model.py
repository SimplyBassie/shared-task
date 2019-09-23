import csv
import pandas as pd

def read_data():
    documents = []
    labels = []
    # ID	INSTANCE	SUBA	SUBB	SUBC
    with open('../Data/olid-training-v1.0.tsv') as tsvfile:
      file = csv.reader(tsvfile, delimiter='\t')
      df = pd.DataFrame(file, columns=['id', 'tweet', 'subtask_a', 'subtask_b', 'subtask_c'])

    return df

def main():
    dataframe = read_data()
    print(dataframe)

if __name__ == '__main__':
    main()
