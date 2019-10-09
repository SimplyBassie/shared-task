import pandas as pd

def main():
    training_data = pd.read_csv("../Data/olid-training-v1.0.tsv", sep='\t')
    print(training_data)


if __name__ == '__main__':
    main()
