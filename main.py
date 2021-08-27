import time
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup, AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from config import Hyper, Constants
from helper import Helper
from data_input import get_datasets
from checkpoint import save_checkpoint, load_checkpoint
from model import training, validation, test_model_for_metrics


def main():
    Helper.printline("** Started **")
    Hyper.start()
    #---------- DATA -------------# 
    train_dataset, val_dataset, test_dataset, country_key, country_label_list = get_datasets()
    
    train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = Hyper.batch_size # Trains with this batch size.
        )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
                val_dataset, # The validation samples.
                sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                batch_size = Hyper.batch_size # Evaluate with this batch size.
            )
    
    # Load BertForSequenceClassification, the pretrained BERT model with a single 
    # linear classification layer on top. 
    model = BertForSequenceClassification.from_pretrained(
        Hyper.model_name,               # Use the 12-layer BERT model, with a cased vocab.
        num_labels = Hyper.num_labels,  # Labels are either positive or negative sentiment.   
        output_attentions = False,      # Do not return attentions weights.
        output_hidden_states = False,   # Do not return all hidden-states.
    )

    # Tell pytorch to run this model on the GPU.
    model.cuda()
    show_model_stats(model)
    
    # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(),
                  lr = Hyper.learning_rate, # args.learning_rate - default is 5e-5, in the Hyper class we use 2e-5
                  eps = Hyper.eps           # args.adam_epsilon  - default is 1e-8 which we also use.
                )
    # Total number of training steps is [number of batches] x [number of epochs]. 
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * Hyper.total_epochs
    
    # Create a schedule with a learning rate (lr) that decreases linearly from the initial lr 
    # set in the optimizer to 0, after a warmup period during which it increases linearly 
    # from 0 to the initial lr set in the optimizer 
    # (see https://huggingface.co/transformers/main_classes/optimizer_schedules.html).
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)
    
    Helper.set_seed()
    # We'll store a number of quantities such as training and validation loss, 
    # validation accuracy, and timings.
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # For each epoch...
    for epoch_i in range(0, Hyper.total_epochs):
        epoch = epoch_i + 1
        #---------- TRAINING -------------#
        avg_train_loss, training_time, model = training(train_dataloader, model, optimizer, scheduler, epoch)
        #---------- VALIDATION -------------#    
        loss, training_stats = validation(validation_dataloader, model, training_stats, epoch, avg_train_loss, training_time)
        #---------- SAVE MODEL -------------#
        # Epoch completed, save model
        checkpoint = {'epoch': epoch_i, 
            "model_state_dict": model.state_dict(), 
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss}
        save_checkpoint(checkpoint)
    
    Helper.time_lapse(total_t0)
    Helper.printline("Total training took {total_time}") 
    #---------- TESTING AND METRICS -------------#  
    test_model_for_metrics(test_dataset, country_key, country_label_list, model, training_stats)

    Helper.printlines("** Ended **", 2)

def show_model_stats(model):
    # Get all of the model's parameters as a list of tuples.
    params = list(model.named_parameters())
    Helper.printline(f'The BERT model has {len(params)} different named parameters.\n')
    Helper.printline('==== Embedding Layer ====\n')
    for p in params[0:5]:
        show_first_2_parameters(p)

    Helper.printlines('==== First Transformer ====\n', 2)
    for p in params[5:21]:
        show_first_2_parameters(p)

    Helper.printlines('==== Output Layer ====\n', 2)
    for p in params[-4:]:
        show_first_2_parameters(p)
        
def show_first_2_parameters(p):
    p0 = p[0]
    p1 = str(tuple(p[1].size()))
    Helper.printline(f"{p0} {p1}")

if __name__ == "__main__":
    main()