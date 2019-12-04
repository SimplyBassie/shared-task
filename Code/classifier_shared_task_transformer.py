from simpletransformers.classification import ClassificationModel
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import shutil
import os

def main():
	lm_list = [("roberta","roberta-base"), ("xlnet","xlnet-base-cased")]
	subtasks = ["A","B","C"]
	size = "small" # small or large
	for subtask in subtasks:
		for lm in lm_list:
			architecture, model_type = lm[0], lm[1]
			use_language_model(subtask, size, architecture, model_type)

def use_language_model(subtask, size, architecture, model_type):

	path = 'data_SharedTask/subtask'+subtask+'/'+size

	args = {
	  'output_dir': architecture+'/'+model_type+'/'+size+'/subtask'+subtask,
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

	  'overwrite_output_dir': False,
	  'reprocess_input_data': False,
	  
	}

	# Create train_df
	datafile_train = pd.read_csv(path+'/train.tsv', sep='\t', names=["id","labels","subtask","text"])
	train_df = datafile_train[["text","labels"]]
	train_df = train_df[(train_df.index < np.percentile(train_df.index, 1))] #for testing

	# Create eval_df
	datafile_dev = pd.read_csv(path+'/dev.tsv', sep='\t', names=["id","labels","subtask","text"])
	eval_df = datafile_dev[["text","labels"]]
	eval_df = eval_df[(eval_df.index < np.percentile(eval_df.index, 1))] #for testing

	# Create a ClassificationModel
	if 'subtaskc' in path.lower():
		model = ClassificationModel(architecture, model_type, args=args, num_labels=3, use_cuda = False) # You can set class weights by using the optional weight argument
	else:
		model = ClassificationModel(architecture, model_type, args=args, use_cuda = False) # You can set class weights by using the optional weight argument

	# Train the model
	model.train_model(train_df)

	# Evaluate the model
	result, model_outputs, wrong_predictions = model.eval_model(eval_df, cr=classification_report, cm=confusion_matrix)

	print(result['cr']) # Classification report
	print(result['cm']) # Confusion matrix

	try:
		shutil.rmtree('cache')
		shutil.rmtree('runs')
		os.remove(architecture+'/'+model_type+'/'+size+'/subtask'+subtask+'/added_tokens.json')
		os.remove(architecture+'/'+model_type+'/'+size+'/subtask'+subtask+'/config.json')
		os.remove(architecture+'/'+model_type+'/'+size+'/subtask'+subtask+'/pytorch_model.bin')
		os.remove(architecture+'/'+model_type+'/'+size+'/subtask'+subtask+'/special_tokens_map.json')
		os.remove(architecture+'/'+model_type+'/'+size+'/subtask'+subtask+'/tokenizer_config.json')
		os.remove(architecture+'/'+model_type+'/'+size+'/subtask'+subtask+'/training_args.bin')
		os.remove(architecture+'/'+model_type+'/'+size+'/subtask'+subtask+'/vocab.json')
	except:
		pass


if __name__ == '__main__':
	main()