#!/usr/bin/env python
from simpletransformers.classification import MultiLabelClassificationModel
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
from emoji_to_words import emoji_to_words
from hashtags import split_hashtags, do_splitting

def main():
    architecture = 'bert'
    model_type = 'bert-base-uncased'
    subtask = 'C'
    size = "small" # small or large
    balance = False
    output_path = 'outputs/'+architecture+'/'+model_type+'/'+size+'/subtask'+subtask
    
    # Change values above, or args

    if subtask.lower() == 'c':
        numlab = 3
        thresholds = [0.2,1,0.3]
    else:
        numlab = 2
        thresholds = [0.15,0.85]

    args = {
      'output_dir': output_path,
      'cache_dir': 'cache/',
      'max_seq_length': 128,
      'train_batch_size': 8,
      'eval_batch_size': 8,
      'gradient_accumulation_steps': 1,
      'num_train_epochs': 1,
      'weight_decay': 0,
      'learning_rate': 4e-5,
      'adam_epsilon': 1e-8,
      'warmup_ratio': 0.06,
      'warmup_steps': 0,
      'max_grad_norm': 1.0,

      'logging_steps': 50,
      'evaluate_during_training': False,
      'save_steps': 2000,
      'eval_all_checkpoints': True,
      'use_tensorboard': True,

      'overwrite_output_dir': True,
      'reprocess_input_data': False,

      "threshold":thresholds,

    }


    # Create train_df
    if balance:
        path = 'data_balanced_SharedTask/subtask'+subtask+'/'+size
        datafile_train = pd.read_csv(path+'/train_balanced.tsv', sep='\t', names=["id","labels","subtask","text"])
    else:
        path = 'data_SharedTask/subtask'+subtask+'/'+size
        datafile_train = pd.read_csv(path+'/train.tsv', sep='\t', names=["id","labels","subtask","text"])

    train_df = datafile_train[["text","labels"]]
    #train_df = train_df[(train_df.index < np.percentile(train_df.index, 40))]

    train_df['text'] = train_df['text'].apply(lambda x: do_splitting(x))
    train_df['text'] = train_df['text'].apply(lambda x: emoji_to_words(x))

    subtask_c_dict = {0: [1,0,0], 1 : [0,1,0], 2: [0,0,1]}
    subtask_b_dict = {0: [1,0], 1 : [0,1]}
    if subtask.lower() == 'c':
        train_df['labels'] = train_df['labels'].apply(lambda x: subtask_c_dict[x])
    else:
        train_df['labels'] = train_df['labels'].apply(lambda x: subtask_b_dict[x])

    # Create eval_df
    if balance:
        datafile_dev = pd.read_csv(path+'/dev_balanced.tsv', sep='\t', names=["id","labels","subtask","text"])
    else:
        datafile_dev = pd.read_csv(path+'/dev.tsv', sep='\t', names=["id","labels","subtask","text"])

    eval_df = datafile_dev[["text","labels"]]
    #eval_df = eval_df[(eval_df.index < np.percentile(eval_df.index, 40))]
    labels = eval_df["labels"].tolist()

    eval_df['text'] = eval_df['text'].apply(lambda x: do_splitting(x))
    eval_df['text'] = eval_df['text'].apply(lambda x: emoji_to_words(x))

    if subtask.lower() == 'c':
        eval_df['labels'] = eval_df['labels'].apply(lambda x: subtask_c_dict[x])
    else:
        eval_df['labels'] = eval_df['labels'].apply(lambda x: subtask_b_dict[x])

    # Create a MultiLabelClassificationModel
    model = MultiLabelClassificationModel(architecture, model_type, num_labels=numlab, use_cuda = False, args=args) # You can set class weights by using the optional weight argument

    # Train the model
    model.train_model(train_df)
    print("TRAINING DONE")
    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(eval_df)
    print(result)
    print(model_outputs)
    print("EVALUATING DONE")
    # Predict
    predictions, raw_outputs = model.predict(eval_df['text'])
    print(raw_outputs)
    print("PREDICTING DONE")
    for num, pred in enumerate(predictions):
        if pred[0] == 1:
            predictions[num] = 0
        elif pred[1] == 1:
            predictions[num] = 1
        elif subtask.lower() == 'c':
            if pred[2] == 1:
                predictions[num] = 2
            else:
                predictions[num] = 1
        else:
            predictions[num] = 1 #Largest group

    print(classification_report(labels, predictions))
    print(confusion_matrix(labels, predictions))


if __name__ == '__main__':
    main()