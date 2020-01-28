'''
Created on October, 2018
by: Meshal
'''

import numpy as np
import tensorflow as tf
import scipy.io
from CATA import CAttAE, parameters
import argparse
import h5py

###################################################
# Load all needed data files
def load_data(sparse, data_name):
    data = {}
    if (data_name == 'a'):
        variables = scipy.io.loadmat(data_dir + "mult_nor.mat")
        data["content"] = variables['X']
    elif (data_name == 't'):
        f = h5py.File(data_dir + "mult_nor.mat",'r') 
        d = f.get('X')
        d = np.array(d).T
        data["content"] = d
    
    if (sparse):
        for data_num in range(1,4):
            data["train_users"+str(data_num)] = load_file(data_dir+str(data_num)+"/cf-train-1-users.dat")
            data["train_items"+str(data_num)] = load_file(data_dir+str(data_num)+"/cf-train-1-items.dat")
            data["test_users"+str(data_num)] = load_file(data_dir+str(data_num)+"/cf-test-1-users.dat")
    else:
        for data_num in range(1,4):
            data["train_users"+str(data_num)] = load_file(data_dir+str(data_num)+"/cf-train-10-users.dat")
            data["train_items"+str(data_num)] = load_file(data_dir+str(data_num)+"/cf-train-10-items.dat")
            data["test_users"+str(data_num)] = load_file(data_dir+str(data_num)+"/cf-test-10-users.dat")
            
    return data

def load_file(path):
    arr = []
    for line in open(path):
        a = line.strip().split()
        if a[0]==0:
            l = []
        else:
            l = [int(x) for x in a[1:]]
        arr.append(l)
    return arr
###################################################
# parse boolean argument
def str2bool(a):
    if a.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif a.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
###################################################
# Main
###################################################
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_name", type=str, help="Dataset name: a or t (default: a)", default='a')
parser.add_argument("-s", "--sparse", type=str2bool, help="Sparse or Dense option (default: sparse)", default=True)
parser.add_argument("-pretrain", "--pretrain", type=str2bool, help="Pretrain the attentive autoencoder? (default: No)", default=False)
parser.add_argument("-o", "--output_name", type=str, help="CATA.mat name",  default='CATA')
parser.add_argument("-e", "--epochs", type=int, help="Number of epochs to pretrain the attentive autoencoder (default: 150)", default=150)
parser.add_argument("-pe", "--pmf_epochs", type=int, help="Number of epochs for pmf (default: 100)", default=100)
parser.add_argument("-l", "--latent_size", type=int, help="Size of latent space (default: 50)", default=50)
parser.add_argument("-u", "--lambda_u", type=float, help="Value of user regularizer (default: 10)", default=10)
parser.add_argument("-v", "--lambda_v", type=float, help="Value of item regularizer (default: 0.1)", default=0.1)

args = parser.parse_args()
np.random.seed(0)
tf.set_random_seed(0)

sparse = args.sparse
data_name = args.data_name
output_name = args.output_name
pretrain = args.pretrain
pretraining_epochs = args.epochs
pmf_epochs = args.pmf_epochs
lambda_u = args.lambda_u
lambda_v = args.lambda_v
latent_size = args.latent_size

print ("=======================================")
print ("               CATA                    ")
print ("=======================================")
if (data_name == 'a'):
    num_users = 5551
    num_items = 16980
    data_dir = "data/citeulike-a/"
    print ("Dataset name: Citeulike-a    #users: 5551    #items:16980")
    print ("Sparse settings? ", sparse)
    print ("Output file name? ", output_name)
elif (data_name == 't'):
    num_users = 7947
    num_items = 25975
    data_dir = "data/citeulike-t/"
    print ("Dataset name: Citeulike-t    #users: 7947    #items:25975")   
    print ("Sparse settings,p=1? ", sparse) 
    print ("Output file name? ", output_name) 

data = load_data(sparse, data_name)
input_size = len(data["content"][0])
print ("Input Dimension: ", input_size)
print ("pre-train? ", pretrain)
print ("pre-training epochs: ", pretraining_epochs)
print ("pmf epochs: ", pmf_epochs)
print ("lambda u: ", lambda_u, "lambda v: ", lambda_v)
print ("Size of latent space:", latent_size)
print ("---------------------------------------")

parameters(num_users, num_items, pmf_epochs, pretraining_epochs, lambda_u, lambda_v, latent_size)
CAttAE(X=data["content"], data=data, sparse=sparse, data_name=data_name, output_name=output_name, pretrain=pretrain)
