import dgl
import dgl.function as fn
from dgl import DGLGraph
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import os
import numpy as np
from numpy import genfromtxt

import csv
from shutil import copyfile
from GCN import HSModel, read_edge, read_node, evaluate

dataset_dir = "./newlayout/newdata/"

MaxMinValues = genfromtxt(dataset_dir + 'MaxMinValues.csv', delimiter=',')

tPower_max = MaxMinValues[1, 0]
tPower_min = MaxMinValues[1, 1]
Power_max = MaxMinValues[1, 2]
Power_min = MaxMinValues[1, 3]
Temperature_max = MaxMinValues[1, 4]
Temperature_min = MaxMinValues[1, 5]
Conductance_max = MaxMinValues[1, 6]
Conductance_min = MaxMinValues[1, 7]

print(tPower_max, tPower_min, Power_max,Power_min,Temperature_max,Temperature_min,Conductance_max,Conductance_min)

last_saved_epoch = 0
date = '20250311'
dir_name = 'GCN'
ckpt_dir = f'ckpt/Syn_{dir_name}_{date}'

if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)

is_inference = False
ckpt_file = ckpt_dir + '/HSgcn_26.pkl'

n_hidden_n = [1, 16, 32, 64, 128, 256, 512, 512, 512, 256, 128, 64, 32, 16, 1]#[1, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512,512,  1024,  512,512, 256,256,128, 128, 64, 64]
e_hidden_e = [1, 16, 32, 64, 128, 256, 512, 512, 512, 256, 128, 64, 32, 16, 0]#[1, 16, 32,32,  64,64, 128,128, 256,256, 512,512,  1024,  512,512, 256,256,128, 128, 64, 0]

batch_size = 1

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Running on GPU!')
else:
    device = torch.device('cpu')
    print('Runing on CPU!')


def dataset(dataset_dir: str, training_data_file: str = 'train_data.csv', test_data_file: str = 'test_data.csv'):
    reader = csv.reader(open(dataset_dir + training_data_file, "r"), delimiter=",")
    train_data = list(reader)
    train_data = np.array(train_data)
    train_data = np.random.permutation(train_data)
    train_data = train_data.reshape(-1).tolist()
    num_train = len(train_data)

    reader = csv.reader(open(dataset_dir + test_data_file, "r"), delimiter=",")
    test_data = list(reader)
    test_data = np.array(test_data)
    test_data = np.random.permutation(test_data)
    test_data = test_data.reshape(-1).tolist()
    num_test = len(test_data)

    return train_data, test_data, num_train, num_test


def loss_function(loss: str = "MSE"):
    if loss == "MSE":
        MSEloss = nn.MSELoss()
        return MSEloss
    else:
        raise NotImplementedError
    

def fine_tuning(model_path: str, model_name: str, model: HSModel, tuning_epoch: int, num_tunable_layer: int):

    # Define dataset for training
    train_data, test_data, num_train, num_test = dataset(dataset_dir=dataset_dir)
    Test_Acc_min = 1
    last_saved_epoch = 0

    print(f'batch size: {batch_size}, num_train: {num_train}')

    # Define loss function and optimizer
    MSEloss = loss_function(loss="MSE")
    select_param = []
    for module in list(model.children())[-1 * num_tunable_layer:]:
        select_param.extend(module.parameters())
    
    optimizer = torch.optim.Adam(
        params=select_param, 
        lr=5e-5
    )

    # Load pretrianed model weights
    model.load_state_dict(torch.load(f"{model_path}/{model_name}", map_location=device))
    model.to(device=device)

    dur=[0]
    # Fine-tuning using constrained layout for tuning_epoch iterations
    for epoch in range(tuning_epoch):
        train_data = np.array(train_data)
        train_data = np.random.permutation(train_data)
        train_data = train_data.reshape(-1).tolist()

        model.train()

        epoch_loss_spec = 0
        epoch_acc_spec = 0
        epoch_MAE_spec = 0
        epoch_AEmax_spec = 0

        if epoch >= 3:
            t0 = time.time()

        for i in range(0, num_train, batch_size):  
            train_batch = train_data[i : min(i + batch_size, num_train)]
            g, edge_feats = read_edge(dataset_dir=dataset_dir, train_batch=train_batch)
            node_feats1, node_feats2, node_labels = read_node(dataset_dir=dataset_dir, train_batch=train_batch)

            g = g.to(device=device)
            node_feats1 = torch.Tensor(node_feats1).to(device=device)
            node_feats2 = torch.Tensor(node_feats2).to(device=device)
            node_labels = torch.Tensor(node_labels).to(device=device)
            edge_feats = torch.Tensor(edge_feats).to(device=device)

            hv = model(g, node_feats2, node_feats1, edge_feats)

            loss = MSEloss(hv, node_labels)
                
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            acc_current, node_output, MAE, AEmax, _ = evaluate(model, g, node_feats1, node_feats2, node_labels, edge_feats)
            epoch_acc_spec += acc_current
            epoch_loss_spec += loss.item()
            epoch_MAE_spec += MAE
            if epoch_AEmax_spec < AEmax:
                epoch_AEmax_spec = AEmax

            if i + 2 * batch_size >= num_train and i + batch_size < num_train:
                length_pred = int(math.sqrt((node_output.shape[0]/batch_size - 12)/3))
                length_stan = int(math.sqrt((node_labels.shape[0]/batch_size - 12)/3))
                for t in range(1):
                    with open(dataset_dir+'Chiplet.grid.steady','w') as grid:
                        for m in range(length_pred):
                            for n in range(length_pred):
                                grid.write(str(m*length_pred+n)+" "+ str((node_output[m*length_pred+n][0].tolist()+1)/2.0*(Temperature_max -Temperature_min)+Temperature_min)+"\n")
                            grid.write("\n")
                    cmd = "../grid_thermal_map.pl Chiplet_Core"+ train_batch[t][:train_batch[t].find('_')]+".flp "+dataset_dir+"Chiplet.grid.steady "+str(length_pred)+" "+ str(length_pred)+" > "+dataset_dir+"train_Chiplet_pred"+str(epoch)+"_"+str(0)+".svg"
            
                    os.system(cmd)
    
                    with open(dataset_dir+'Chiplet.grid.steady','w') as grid:
                        for m in range(length_stan):
                            for n in range(length_stan):
                                grid.write(str(m*length_stan+n)+" "+ str((node_labels[m*length_stan+n][0].tolist()+1)/2.0*(Temperature_max -Temperature_min)+Temperature_min)+"\n")
                            grid.write("\n")
                    cmd = "../grid_thermal_map.pl Chiplet_Core"+ train_batch[t][:train_batch[t].find('_')]+".flp "+dataset_dir+"Chiplet.grid.steady "+str(length_stan)+" "+ str(length_stan)+" > "+dataset_dir+"train_Chiplet_stan"+str(epoch)+"_"+str(0)+".svg"
            
                    os.system(cmd)

            if i % 50 == 0:
                print(f"epoch: {epoch}, i: {i}")
                
            if epoch >= 3:
                dur.append(time.time() - t0)

        epoch_acc_spec /= int(num_train / batch_size) + (num_train % batch_size > 0)
        epoch_loss_spec /= int(num_train / batch_size) + (num_train % batch_size > 0)
        epoch_MAE_spec /= int(num_train / batch_size) + (num_train % batch_size>0)
            
            # print(f"Train Epoch {epoch} |Time(s) {np.mean(dur)} | Loss {epoch_loss} | Accuracy {epoch_acc} | MAE {epoch_MAE * (Temperature_max - Temperature_min)} | AEmax {epoch_AEmax*(Temperature_max - Temperature_min)}")
        with open(dataset_dir + 'train_acc_GCN_tun.txt','a') as Train_Acc_file:
            Train_Acc_file.write(f"epoch: {epoch}, epoch_loss: {epoch_loss_spec}, epoch_acc: {epoch_acc_spec}, MAE: {epoch_MAE_spec * (Temperature_max - Temperature_min)}, AEmax: {epoch_AEmax_spec * (Temperature_max - Temperature_min)}\n")

        if epoch % 1 == 0:
            epoch_test_acc = 0
            epoch_test_MAE = 0
            epoch_test_AEmax = 0
            for i in range(0, num_test, batch_size):  
                test_batch = test_data[i : min(i+batch_size, num_test)]
                g, edge_feats = read_edge(dataset_dir=dataset_dir, train_batch=test_batch)
                node_feats1, node_feats2, node_labels = read_node(dataset_dir=dataset_dir, train_batch=test_batch)

                g = g.to(device=device)
                node_feats1 = torch.Tensor(node_feats1).to(device=device)
                node_feats2 = torch.Tensor(node_feats2).to(device=device)
                node_labels = torch.Tensor(node_labels).to(device=device)
                edge_feats = torch.Tensor(edge_feats).to(device=device)

                acc, node_output, MAE, AEmax, _ = evaluate(model, g, node_feats1, node_feats2, node_labels, edge_feats)
                    
                epoch_test_acc += acc
                epoch_test_MAE += MAE
                if epoch_test_AEmax < AEmax:
                    epoch_test_AEmax = AEmax

                if i+2*batch_size >= num_test and i+batch_size < num_test:
                    with open(dataset_dir+'Chiplet.grid.steady','w') as grid:
                        for m in range(length_pred):
                            for n in range(length_pred):
                                grid.write(str(m*length_pred+n)+" "+ str((node_output[m*length_pred+n][0].tolist()+1)/2.0*(Temperature_max -Temperature_min)+Temperature_min)+"\n")
                            grid.write("\n")
                    cmd = "../grid_thermal_map.pl Chiplet_Core"+ test_batch[0][:test_batch[0].find('_')]+".flp "+dataset_dir+"Chiplet.grid.steady "+str(length_pred)+" "+ str(length_pred)+" > "+dataset_dir+"test_Chiplet_pred"+str(epoch)+"_"+str(0)+".svg"
                
                    os.system(cmd)
    
                    with open(dataset_dir+'Chiplet.grid.steady','w') as grid:
                        for m in range(length_stan):
                            for n in range(length_stan):
                                grid.write(str(m*length_stan+n)+" "+ str((node_labels[m*length_stan+n][0].tolist()+1)/2.0*(Temperature_max -Temperature_min)+Temperature_min)+"\n")
                            grid.write("\n")
                    cmd = "../grid_thermal_map.pl Chiplet_Core"+ test_batch[0][:test_batch[0].find('_')]+".flp "+dataset_dir+"Chiplet.grid.steady "+str(length_stan)+" "+ str(length_stan)+" > "+dataset_dir+"test_Chiplet_stan"+str(epoch)+"_"+str(0)+".svg"
                
                    os.system(cmd)

            epoch_test_acc /= int(num_test/batch_size) + (num_test%batch_size > 0)
            epoch_test_MAE /= int(num_test/batch_size) + (num_test % batch_size > 0)

            print(f"Test Epoch {epoch} | Accuracy {epoch_test_acc}")
            with open(dataset_dir + 'train_acc_GCN_tun.txt','a') as Train_Acc_file:
                Train_Acc_file.write(f"epoch: {epoch}, epoch_test_acc {epoch_test_acc}, test_MAE: {epoch_test_MAE*(Temperature_max - Temperature_min)} test_AE: {epoch_test_AEmax*(Temperature_max - Temperature_min)}\n")
            
        # Save the best model according to test
        
        if epoch_test_acc < Test_Acc_min:
            Test_Acc_min = epoch_test_acc
            if os.path.exists(ckpt_dir + f'/HSgcn_{last_saved_epoch}_FT.pkl'):
                os.remove(ckpt_dir + f'/HSgcn_{last_saved_epoch}_FT.pkl')

            torch.save(model.state_dict(), ckpt_dir + f'/HSgcn_{epoch}_FT.pkl')
            print("model saved")
                
            last_saved_epoch = epoch



if __name__ == "__main__":
    fine_tuning(
        model=HSModel(1, n_hidden_n, e_hidden_e, F.relu),
        model_path="ckpt/Syn_GCN_20250311",
        model_name="HSgcn_26.pkl",
        tuning_epoch=20,
        num_tunable_layer=10
    )