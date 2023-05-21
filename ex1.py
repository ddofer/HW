# -*- coding: utf-8 -*-
## __author__ : Dan
import argparse
import torch
import time
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
# import wandb

def main(args):
    # wandb.login()
    models = ['roberta-base','bert-base-uncased', 'google/electra-base-generator'] # what, no deberta-V3? (the SOTA barring feat.engineering :D)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = load_dataset('glue', 'sst2')

    if args.train_samples != -1:
        dataset['train'] = dataset['train'].select(range(args.train_samples))

    if args.val_samples != -1:
        dataset['validation'] = dataset['validation'].select(range(args.val_samples))

    if args.test_samples != -1:
        dataset['test'] = dataset['test'].select(range(args.test_samples))
    
    mean_std_accuracies = []
    total_train_time = 0# save/get total train time of all models
    
    for model_name in models:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        
        def preprocess_function(examples):
            return tokenizer(examples['sentence'], truncation=True, padding=True, max_length=config.max_position_embeddings-2)#'longest')#, max_length=config.max_position_embeddings-2) - mlen autoloaded

        encoded_dataset = dataset.map(preprocess_function, batched=True)

        seeds = list(range(args.seeds))
        accuracies = []

        for seed in seeds:
            model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config).to(device) # dro pto device?
            training_args = TrainingArguments(
                output_dir='./results',
                overwrite_output_dir=True,
                # num_train_epochs= 1,#3, # use default
                logging_dir='./logs',
                seed=seed,
                # report_to="wandb",
                logging_strategy="epoch",
                evaluation_strategy="epoch",
                run_name=f"{model_name}_seed_{seed}",
                metric_for_best_model = "eval_accuracy",
                greater_is_better=True,
                save_strategy="epoch",
                load_best_model_at_end=True,
                save_total_limit = 1,
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=encoded_dataset['train'],
                eval_dataset=encoded_dataset['validation'],
                tokenizer=tokenizer,
                compute_metrics = lambda x: {'accuracy': (x.predictions.argmax(-1) == x.label_ids).mean()}
            )
            trainer.train()
            total_train_time += trainer.state.log_history[-1]['train_runtime']
            evaluation_output = trainer.evaluate()

            accuracies.append(evaluation_output['eval_accuracy'])

        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)

        mean_std_accuracies.append((mean_accuracy, std_accuracy, model_name))

    best_model = max(mean_std_accuracies, key=lambda x: x[0]) # this is max mean acc 
    best_model_name = best_model[2]
    best_seed = seeds[np.argmax(accuracies)]
    trainer.save_model('best_model') # save trainer's best model + weights
    print("mean_std_accuracies\n",mean_std_accuracies)
    print("best_model\n",best_model)

    with open("res.txt", "w") as f:
        for model_result in mean_std_accuracies:
            f.write(f"{model_result[2]},{model_result[0]} +- {model_result[1]}\n") 
        ### https://stackoverflow.com/questions/68806265/huggingface-trainer-logging-train-data 
        f.write("----\n")
        
        f.write(f"train time,{total_train_time}\n")

    # #######################
    # print("best_model_name",best_model_name)
    # print("best_seed",best_seed)
    # print("mean acc",mean_accuracy)
    # print("max val acc", np.max(accuracies))
    config = AutoConfig.from_pretrained("best_model")#best_model_name) # added 
    ### load best, trained model from local disk/checkpoint
    best_model = AutoModelForSequenceClassification.from_pretrained( "best_model"#best_model_name,
                                                                    ,config=config).to(device)
    best_tokenizer = AutoTokenizer.from_pretrained("best_model")#best_model_name)

    ### NOTE: instead of proceeding with the trainer, we could save the best model to disk and reload it as a model; but this also works

    def predict_preprocess_function(examples):
        """Test/inference - don't pad"""
        return best_tokenizer(examples['sentence'], truncation=True, max_length=config.max_position_embeddings-2)#True)#, padding='longest', max_length=config.max_position_embeddings-2)#, padding="do_not_pad") # change padding to none

    best_model.eval()
    predictions = []

    test_dataset = dataset['test'].map(predict_preprocess_function, batched=False,batch_size=1) # tokenize in adv. no truncate
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    ## Probably unneeded, since trainer is set to load best model at end of training, but this ensures it's the chosen by mean acc, and not just by best (max) val_acc  
    trainer = Trainer(
        model=best_model,
        model_init="best_model",
        args= TrainingArguments(seed=best_seed,output_dir='./results',overwrite_output_dir=True,
                report_to="wandb"),
        tokenizer=best_tokenizer
    )

    # Evaluate on test set
    ## https://huggingface.co/learn/nlp-course/chapter3/4?fw=pt 
        ## Uses best model from train/eval in trainer. 

    test_output = trainer.predict(test_dataset.remove_columns("label"))## ORIG#  # test dataset ahas -1 as labels ; was causing cuda side assert error (when using trainre)

    test_runtime = test_output.metrics["test_runtime"] # prediction time in seconds
    with open("res.txt", "a") as f:
        ### check thatt his isn't wiped
        # f.write(f"train time,{trainer.state.log_history[-2]['train_runtime']}\n") ## NOTE: check that this is time of best model and not totla models or last model??
        f.write(f"predict time,{test_runtime}")

    predictions = test_output.predictions.argmax(-1)

    with open("predictions.txt", "w") as f:
        for i,pred in enumerate(predictions):
            f.write(f"{dataset['test']['sentence'][i]}###{pred}\n")

# ## for running interactively in notebook:
# class Args:
#     def __init__(self, seeds, train_samples, val_samples, test_samples):
#         self.seeds = seeds
#         self.train_samples = train_samples
#         self.val_samples = val_samples
#         self.test_samples = test_samples

# # args = Args(2,256,64,16)
# args = Args(3,-1,-1,-1)
# ## note: SOTA on dev set acc is ~ 96 (with bigger models and more epochs etc'. I get good/comparable with my own models - ST or SB autoML)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("seeds", type=int, help="Number of seeds to be used for each model")
    parser.add_argument("train_samples", type=int, help="Number of samples to be used during training or -1 if all training samples should be used")
    parser.add_argument("val_samples", type=int, help="Number of samples to be used during validation or -1 if all validation samples should be used")
    parser.add_argument("test_samples", type=int, help="Number of samples for which the model will predict a sentiment or -1 if sentiment should be predicted for all test samples")
    
    args = parser.parse_args()
    main(args)

