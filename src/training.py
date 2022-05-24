import torch
from tqdm import tqdm
import numpy as np
import math
from pytorchtools import EarlyStopping

def train(model, optimizer, loss_fn, train_dl, val_dl, epochs=100, device='cpu'):
    model = model.to(device)
    
    early_stopping = EarlyStopping(patience=10, verbose=True)
    
    print('train(): model=%s, opt=%s(lr=%f), epochs=%d, device=%s\n' % \
          (type(model).__name__, type(optimizer).__name__,
           optimizer.param_groups[0]['lr'], epochs, device))
    
    loss_fn_red=torch.nn.CrossEntropyLoss(reduction=none)
    
    history = {}
    history['train_acc'] = []
    history['true_train_acc'] = []
    history['val_acc'] = []
    history['distance'] = []
    history['loss'] = []
    history['loss_clean'] = []
    
    W_0=[]
    for layer in model.children():
        try:
            weights = layer.weight.data
            W_0.append(weights)
        except AttributeError:
            pass

    pbar = tqdm(range(1, epochs+1))
    for _ in pbar:
        model.train()
        num_train_correct = 0
        num_train_correct_true = 0
        num_train_examples = 0
        running_loss = 0.0

        
        for batch in train_dl:
            optimizer.zero_grad()

            x = batch[0].to(device)
            y = batch[1].to(device)
            y_true = batch[2].to(device)
            yhat = model(x)
         
            if y==y_true:
                loss = loss_fn_red(yhat, y)
                loss_clean=loss.cpu().detach().numpy()
                history['loss_clean'].append(loss_clean)
            else:
                loss = loss_fn_red(yhat, y_true)
                loss_cor=loss.cpu().detach().numpy()
                history['loss'].append(loss_cor)
           
            loss = loss_fn(yhat, y)
            loss.backward()
            optimizer.step()

            running_loss+=loss
            num_train_correct += (torch.max(yhat, 1)[1] == y).sum().item()
            num_train_correct_true += (torch.max(yhat, 1)[1] == y_true).sum().item()
            num_train_examples += x.shape[0]

     

        W_T=[]
        for layer in model.children():
            try:
                weights = layer.weight.data
                W_T.append(weights)
            except AttributeError:
                pass
        

        train_acc = num_train_correct / num_train_examples
        true_train_acc = num_train_correct_true / num_train_examples

        model.eval()
        num_val_correct  = 0
        num_val_examples = 0

        for batch in val_dl:
            x = batch[0].to(device)
            y = batch[1].to(device)
            yhat = model(x)
            loss = loss_fn(yhat, y)

            num_val_correct += (torch.max(yhat, 1)[1] == y).sum().item()
            num_val_examples += y.shape[0]

        val_acc = num_val_correct / num_val_examples

        distance=0
        W_0 = torch.cat([w.view(-1) for w in W_0])
        W_T = torch.cat([w.view(-1) for w in W_T])
     
        distance=np.linalg.norm(W_0.cpu()-W_T.cpu())

        pbar.set_description('train acc: %5.2f, true train acc: %5.2f, val acc: %5.2f' % (train_acc, true_train_acc, val_acc))

        history['train_acc'].append(train_acc)
        history['true_train_acc'].append(true_train_acc)
        history['val_acc'].append(val_acc)

        history['distance'].append(distance)
        #history['loss'].append(math.sqrunning_loss))
        
        early_stopping(loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            #break

    return history
