import numpy as np
import numpy.matlib
import torch
import torch.nn as nn


def get_Jacobian_svd(model, dl_train):
    device = torch.device('cuda')
    num_classes=10
    grad_batch = []
    i=0
    for batch in dl_train:
        
        i+=1
        cur_gradient = []
        x = batch[0].to(device)
        y = batch[1].to(device)
        for cur_lbl in range(1):
            model.zero_grad()
            cur_output = model(x)
            #print(np.shape(cur_output))
            cur_one_hot = [0] * int(num_classes)
            cur_one_hot[cur_lbl] = 1
            cur_one_hot=np.matlib.repmat([cur_one_hot], np.shape(cur_output)[0], 1)
            #print(np.shape(cur_one_hot))
            cur_one_hot = torch.FloatTensor(cur_one_hot).cuda()

            #model.zero_grad()
            #cur_output = model(x)
            cur_output.backward(cur_one_hot)
            for para in model.parameters():
                cur_gradient.append(para.grad.data.cpu().numpy().flatten())
        
        grad_batch.append(np.concatenate(cur_gradient)) 


    uv, sv, vtv = np.linalg.svd(grad_batch, full_matrices=False) 
    
    return sv