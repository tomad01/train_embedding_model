import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os, json
from sklearn.metrics import confusion_matrix,accuracy_score
import numpy as np
from tqdm import tqdm
from functools import partial

def get_device():
    # Check if mps is available
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return  'cuda'
    return 'cpu'

def find_most_confused_pairs(confusion_matrix):
    # Create a list to hold non-diagonal errors (pair and value)
    errors = []
    # Iterate over the confusion matrix to extract off-diagonal elements
    labels = np.arange(confusion_matrix.shape[0])
    for i,class_x in enumerate(labels):
        for j,class_y in enumerate(labels):
            if i != j:
                errors.append(((class_x, class_y), confusion_matrix[i, j]))

    # Sort the errors in descending order by the error count
    errors_sorted = sorted(errors, key=lambda x: x[1], reverse=True)
    return errors_sorted

def train_head_model(
    head_model,
    train_dataset,
    test_dataset,
    epochs=10,
    learning_rate=0.001,
    batch_size=32,
    logs_path="head_model_logs"
):
    """
    Trains the head model and returns the most confused classes.
    """
    # Ensure logs path exists
    os.makedirs(logs_path, exist_ok=True)
    
    # head_model.half()
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(head_model.parameters(), lr=learning_rate)
    
    # Training loop
    device = get_device()
    head_model.to(device)
    history = {'train_loss': [], 'test_loss': [], 'test_accuracy': [], 'learning_rate': []}

    for _ in tqdm(range(epochs),total=epochs,desc="Training Head Model"):
        head_model.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = head_model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # Final Evaluation
        head_model.eval()
        preds = []
        test_total_loss = 0
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = head_model(X_batch)
            loss = criterion(outputs, y_batch)
            test_total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            preds.extend(predictions)
        y_true = test_dataset.labels.cpu().numpy().tolist()
        accuracy = accuracy_score(y_true,preds)

        history['train_loss'].append(total_loss / len(train_loader))
        history['test_loss'].append(test_total_loss / len(test_loader))
        history['test_accuracy'].append(accuracy)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        with open(os.path.join(logs_path, "history.json"), "w") as f:
            json.dump(history, f)

        

    # save model
    torch.save(head_model.state_dict(), os.path.join(logs_path, "head_model.pth"))
    # Compute confusion matrix
    cmatrix = confusion_matrix(y_true,preds)
    return find_most_confused_pairs(cmatrix)

def triplet_collate_fn(batch, tokenizer):
    anchors, positives, negatives = zip(*batch)
    
    # Tokenize the anchor, positive, and negative texts
    anchor_tokens = tokenizer(
        list(anchors),
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    positive_tokens = tokenizer(
        list(positives),
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    negative_tokens = tokenizer(
        list(negatives),
        padding=True,
        truncation=True,
        return_tensors='pt'
    )

    return anchor_tokens, positive_tokens, negative_tokens

# Training function
def train_body_model(body_model, X_train, X_test, epochs=1,
                     learning_rate=0.00001, batch_size=32, logs_path=None,optimizer=None,history=None):
    os.makedirs(logs_path, exist_ok=True)
    triplet_collate_fn_with_tokenizer = partial(triplet_collate_fn, tokenizer=body_model.tokenizer)
    # Prepare DataLoaders
    train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True,collate_fn=triplet_collate_fn_with_tokenizer)
    test_loader = DataLoader(X_test, batch_size=batch_size, shuffle=False,collate_fn=triplet_collate_fn_with_tokenizer)
    
    # Define optimizer
    # optimizer = optim.Adam(body_model.parameters(), lr=learning_rate)
    
    # Define Triplet Margin Loss
    criterion = nn.TripletMarginLoss(margin=0.7, p=2) # try also a margin of 0.5, 1.0
    
    device = get_device()
    body_model.to(device)
    # Training loop
    body_model.train()
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    # history = {'train_loss': [], 'test_loss': [], 'learning_rate': []}

    for _ in range(epochs):
        loss_acum = []  
        for _, (anchor, positive, negative) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training Body Model"):
            optimizer.zero_grad()
            
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            # Forward pass
            # Enables autocasting for the forward pass (model + loss)
            # with torch.autocast(device_type=device):

            anchor_embed = body_model(anchor)['sentence_embedding']
            positive_embed = body_model(positive)['sentence_embedding']
            negative_embed = body_model(negative)['sentence_embedding']
            # Compute loss
            loss = criterion(anchor_embed, positive_embed, negative_embed)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
        
            loss_acum.append(loss.item())
            if len(loss_acum) == 10:
                loss_avg = sum(loss_acum) / len(loss_acum)
                loss_acum = []
                history['train_loss'].append(loss_avg)
        # scheduler.step()
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])

        # Optionally, evaluate on the test set
        body_model.eval()
        loss_acum = []
        
        with torch.no_grad():
            for anchor, positive, negative in tqdm(test_loader,total=len(test_loader),desc="Evaluating Body Model"):
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                anchor_embed = body_model(anchor)['sentence_embedding']
                positive_embed = body_model(positive)['sentence_embedding']
                negative_embed = body_model(negative)['sentence_embedding']
                
                loss = criterion(anchor_embed, positive_embed, negative_embed)
                loss_acum.append(loss.item())
                if len(loss_acum) == 10:
                    loss_avg = sum(loss_acum) / len(loss_acum)
                    loss_acum = []
                    history['test_loss'].append(loss_avg)

        with open(os.path.join(logs_path, "history.json"), "w") as f:
            json.dump(history, f)

        # Save the model
        # torch.save(body_model.state_dict(), os.path.join(logs_path, "body_model.pth"))
        body_model.save(os.path.join(logs_path, "fine_tuned_body_model"))
            
        return body_model
