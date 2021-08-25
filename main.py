from country import Country
from charts import Chart
from tokens_bert import TokensBert
import pandas as pd
import numpy as np
import os
import torch as T
import time
import random
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup, AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from config import Hyper, Constants
from helper import Helper
from sentiment import Sentiment
from tokens_bert import TokensBert
from pickle_file import Pickle

def main():
    Helper.printline("** Started **")
    Hyper.start()
    file = os.path.join(Constants.HyrdatedTweetLangDir, Constants.HyrdatedLangTweetFile)
    Helper.printline(f"Tweet file: {file}")
    df = pd.read_csv(file, sep=',', error_bad_lines=False, index_col=False, dtype="unicode")
    Helper.printline(f"Before {df.shape[0]}")
    _query = Helper.countries_query_builder()
    df.query(_query, inplace=True)
    Helper.printline(f"After {df.shape[0]}")
    #Chart.show_country_distribution(df)
    X = list(df["clean_text"])
    y = pd.factorize(df['Country'])[0].astype(int)
    #y = list(df["sentiment"].astype(int))
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.20, random_state=1)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=1)
    ''' s = Sentiment(df)
    s.print_balance() '''
    c = Country(df)
    c.print_balance()

    # Load the BERT tokenizer.
    # The Pickle.get_content method loads the content from the pickle file if it exists
    # or otherwise it tokenises the input from the Bert tokeniser and saves the results in the pickle file
    # 
    Helper.printline("Encode training data")
    Helper.printline("--------------------")
    t = TokensBert(X_train)
    X_train_enc = Pickle.get_content(Constants.pickle_train_encodings_file, t.encode_tweets)
   
    Helper.printline("Encode validation data")
    Helper.printline("----------------------")
    t = TokensBert(X_val)
    X_val_enc = Pickle.get_content(Constants.pickle_val_encodings_file, t.encode_tweets)
    train_dataset = get_dataset(y_train, X_train_enc)
    val_dataset = get_dataset(y_val, X_val_enc)
    
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
    
    set_seed()
    # We'll store a number of quantities such as training and validation loss, 
    # validation accuracy, and timings.
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # For each epoch...
    for epoch_i in range(0, Hyper.total_epochs):
        
        # ========================================
        #               Training
        # ========================================
        
        # Perform one full pass over the training set.
        Helper.printline(f'\n======== Epoch {epoch_i + 1} / {Hyper.total_epochs} ========')
        Helper.printline('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Set the model mode to train
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 100 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = Helper.format_time(time.time() - t0)
                
                # Report progress.
                Helper.printline(f'  Batch {step}  of  {len(train_dataloader)}.    Elapsed: {elapsed}.')

            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the 
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].to(Constants.device)
            b_input_mask = batch[1].to(Constants.device)
            b_labels = batch[2].to(Constants.device)
                      
            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because 
            # accumulating the gradients is "convenient while training RNNs". 
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()        

            # Perform a forward pass (evaluate the model on this training batch).
            # In PyTorch, calling `model` will in turn call the model's `forward` 
            # function and pass down the arguments. The `forward` function is 
            # documented here: 
            # https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification
            # The results are returned in a results object, documented here:
            # https://huggingface.co/transformers/main_classes/output.html#transformers.modeling_outputs.SequenceClassifierOutput
            # Specifically, we'll get the loss (because we provided labels) and the
            # "logits"--the model outputs prior to activation.
            result = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask, 
                        labels=b_labels,
                        return_dict=True)

            loss = result.loss
            logits = result.logits

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value 
            # from the tensor.
            _loss = loss.item()
            total_train_loss += _loss

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            T.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)            
        
        # Measure how long this epoch took.
        training_time = Helper.format_time(time.time() - t0)

        Helper.printline("\n  Average training loss: {avg_train_loss}")
        Helper.printline(f"  Training epcoh took: {training_time}")
            
        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        Helper.printline("\nRunning Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables 
        total_eval_accuracy = 0
        total_eval_loss = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            
            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using 
            # the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].to(Constants.device)
            b_input_mask = batch[1].to(Constants.device)
            b_labels = batch[2].to(Constants.device)
            
            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with T.no_grad():        

                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which 
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                result = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask,
                            labels=b_labels,
                            return_dict=True)

            # Get the loss and "logits" output by the model. The "logits" are the 
            # output values prior to applying an activation function like the 
            # softmax.
            loss = result.loss
            logits = result.logits
                
            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += flat_accuracy(logits, label_ids)
            

        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        Helper.printline(f"  Accuracy: {avg_val_accuracy}")

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        
        # Measure how long the validation run took.
        validation_time = Helper.format_time(time.time() - t0)
        
        Helper.printline(f"  Validation Loss: {avg_val_loss}")
        Helper.printline(f"  Validation took: {validation_time}")

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    Helper.printline("Total training took {:} (h:mm:ss)".format(Helper.format_time(time.time()-total_t0)))
    Helper.printline("** Ended **")

def get_dataset(labels, data_enc):
    _inputs = get_tensor(data_enc, "input_ids")
    _masks = get_tensor(data_enc, "attention_mask")
    _labels = T.tensor(labels, dtype=T.long)
    _dataset = TensorDataset(_inputs, _masks, _labels)
    return _dataset

def get_tensor(_tensor, field_name):
    items = [x[field_name] for x in _tensor]
    output = T.cat(items, dim=0)
    return output

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def set_seed():
    random.seed(Constants.seed_val)
    np.random.seed(Constants.seed_val)
    T.manual_seed(Constants.seed_val)
    T.cuda.manual_seed_all(Constants.seed_val)

def show_model_stats(model):
    # Get all of the model's parameters as a list of tuples.
    params = list(model.named_parameters())
    Helper.printline(f'The BERT model has {len(params)} different named parameters.\n')
    Helper.printline('==== Embedding Layer ====\n')
    for p in params[0:5]:
        show_first_2_parameters(p)

    Helper.printline('\n==== First Transformer ====\n')
    for p in params[5:21]:
        show_first_2_parameters(p)

    Helper.printline('\n==== Output Layer ====\n')
    for p in params[-4:]:
        show_first_2_parameters(p)
        
def show_first_2_parameters(p):
    p0 = p[0]
    p1 = str(tuple(p[1].size()))
    Helper.printline(f"{p0} {p1}")

if __name__ == "__main__":
    main()