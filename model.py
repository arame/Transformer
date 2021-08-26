from helper import Helper
from config import Hyper, Constants
import time
import torch as T
import numpy as np
import pandas as pd

def training(train_dataloader, model, optimizer, scheduler, epoch):
    # ========================================
    #               Training
    # ========================================
    
    # Perform one full pass over the training set.
    Helper.printlines(f'======== Epoch {epoch} / {Hyper.total_epochs} ========', 2)
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

    Helper.printlines(f"  Average training loss: {avg_train_loss}", 2)
    Helper.printline(f"  Training epoch took: {training_time}")
    return avg_train_loss, training_time, model

def validation(validation_dataloader, model, training_stats, epoch, avg_train_loss, training_time):
    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    Helper.printlines("Running Validation...", 2)

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
                'epoch': epoch,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )
    
    return loss, training_stats

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
 
def display_training_stats(df_stats):
    # Display floats with two decimal places.
    pd.set_option('precision', 2)

    # Display the table.
    print(df_stats.to_markdown())
