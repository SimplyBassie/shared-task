from simpletransformers.classification import ClassificationModel
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd

def main():
    architecture = 'roberta'
    model_type = 'roberta-base'
    subtask = 'B'
    size = "small" # small or large
    balance = False
    output_path = 'outputs/'+architecture+'/'+model_type+'/'+size+'/subtask'+subtask

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
      
    }

    # Create train_df
    if balance:
        path = 'data_balanced_SharedTask/subtask'+subtask+'/'+size
        datafile_train = pd.read_csv(path+'/train_balanced.tsv', sep='\t', names=["id","labels","subtask","text"])
    else:
        path = 'data_SharedTask/subtask'+subtask+'/'+size
        datafile_train = pd.read_csv(path+'/train.tsv', sep='\t', names=["id","labels","subtask","text"])
    train_df = datafile_train[["text","labels"]]
    train_df = train_df[(train_df.index < np.percentile(train_df.index, 40))] #for testing

    # Create eval_df
    if balance:
        datafile_dev = pd.read_csv(path+'/dev_balanced.tsv', sep='\t', names=["id","labels","subtask","text"])
    else:
        datafile_dev = pd.read_csv(path+'/dev.tsv', sep='\t', names=["id","labels","subtask","text"])
    eval_df = datafile_dev[["text","labels"]]
    eval_df = eval_df[(eval_df.index < np.percentile(eval_df.index, 40))] #for testing

    # Create a ClassificationModel
    if 'subtaskc' in path.lower():
        numlab= 3
        weights = [1,1,1]
    else:
        numlab=2
        weights = [1,1]
    model = ClassificationModel(architecture, model_type, args=args, use_cuda = False, num_labels=numlab, weight=weights) # You can set class weights by using the optional weight argument

    # Train the model
    model.train_model(train_df)

    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(eval_df, cr=classification_report, cm=confusion_matrix)

    print(model_outputs)
    print(result['cr']) # Classification report
    print(result['cm']) # Confusion matrix
    

if __name__ == '__main__':
    main()
