'''
Created on March, 2019
by: Meshal
'''

import scipy
import scipy.io
import numpy as np
from Attentive_autoencoder import AttAE_module, train, get_z_layer, load_model, save_model

# explicit setting
a = 1
b = 0.01

def parameters(num_users, num_items, pmf_epochs, mlp_epoch, lambda_u, lambda_v, latent_size):
    parameters.m_num_users = num_users
    parameters.m_num_items = num_items
    parameters.m_U = 0.1 * np.random.randn(num_users, latent_size)
    parameters.m_V = 0.1 * np.random.randn(num_items, latent_size)
    parameters.m_theta = 0.1 * np.random.randn(num_items, latent_size)
    parameters.m_gamma = 0.1 * np.random.randn(num_items, latent_size)
    parameters.n_epochs = pmf_epochs
    parameters.mlp_epoch = mlp_epoch
    parameters.lambda_u= lambda_u
    parameters.lambda_v= lambda_v
    parameters.dimension= latent_size
    
def restore():
    parameters.m_U = 0.1 * np.random.randn(parameters.m_num_users, parameters.dimension)
    parameters.m_V = 0.1 * np.random.randn(parameters.m_num_items, parameters.dimension)

def evaluate(test_user, M):
	
	ranks = np.arange(1, M+1, 1)
	dcg_ranks = np.arange(2, M+2, 1)
	M = [10, 50,100,150,200,250,300]
	recall = np.zeros((parameters.m_num_users, len(M)))
	mrr = np.zeros((parameters.m_num_users, len(M)))
	dcg = np.zeros((parameters.m_num_users, len(M)))
	dcg_ideal = np.zeros((parameters.m_num_users, len(M)))
	ndcg = np.zeros((parameters.m_num_users, len(M)))
	for i in range(parameters.m_num_users):
		u_tmp = parameters.m_U[i]
		score = np.dot(u_tmp, parameters.m_V.T)
		ind_rec = np.argsort(score)[::-1]
		# construct relevant articles (ground truth)
		gt = np.zeros(parameters.m_num_items)
		for j in (test_user[i]):
    			gt[j] = 1
    		# sort gt according to ind_rec
		gt = gt[ind_rec]
		
		for k in range(len(M)):
    			if (np.sum(gt)==0):
    				recall[i,k] = -1 #Mark rows that sum to 0 --> -1 [those users who have no items to test in test set]; to remove them later
    				mrr[i,k] = -1 
    				dcg[i,k] = -1 
    				dcg_ideal[i,k] = -1 
    				ndcg[i, k] = -1
    			else:
    				recall[i, k] = 1.0*np.sum(gt[:M[k]])/np.sum(gt)
    				first_rank = np.nonzero(gt[:M[k]])
    				mrr[i, k] = (1.0/(first_rank[0][0]+1)) if len(first_rank[0]) > 0 else 0
    				dcg[i, k] = 1.0*np.sum( gt[:M[k]] / np.log2(dcg_ranks[:M[k]]) ) 
    				sorted_gt = np.sort(gt)[::-1]
    				dcg_ideal[i, k] = 1.0*np.sum( sorted_gt[:M[k]] / np.log2(dcg_ranks[:M[k]]) )
    				ndcg[i, k] = 1.0*(dcg[i, k] / dcg_ideal[i, k])
	
	l1 = len(recall)
	recall = recall[np.sum(recall, axis=1)>=0] #remove recall values less tahn 0 [those users who have no items to test in test set]
	mrr = mrr[np.sum(mrr, axis=1)>=0]
	dcg = dcg[np.sum(dcg, axis=1)>=0]
	ndcg = ndcg[np.sum(ndcg, axis=1)>=0]
	l2 = len(recall)
	
	recall = np.mean(recall, axis=0)
	mrr = np.mean(mrr, axis=0)
	dcg = np.mean(dcg, axis=0)
	ndcg = np.mean(ndcg, axis=0)
	return recall, dcg, ndcg

		
def pmf_estimate(users, items, n_epochs):
	a_minus_b = a - b
	iteration = 0
	while (iteration < n_epochs):
		# update U
		# ids for v_j that has at least one user liked
		ids = np.array([len(x) for x in items]) > 0
		vtv = np.dot(parameters.m_V[ids].T, parameters.m_V[ids]) * b
		uu = vtv + np.eye(parameters.dimension) * parameters.lambda_u 
		for i in range(parameters.m_num_users):
			item_ids = users[i]
			n = len(item_ids)
			if n > 0:
				A = uu + np.dot(parameters.m_V[item_ids, :].T, parameters.m_V[item_ids,:])*a_minus_b
				x = a * np.sum(parameters.m_V[item_ids, :], axis=0)  
				parameters.m_U[i, :] = scipy.linalg.solve(A, x) 
		# update V
		ids = np.array([len(x) for x in users]) > 0
		uvu = np.dot(parameters.m_U[ids].T, parameters.m_U[ids]) * b
		vv = uvu + np.eye(parameters.dimension) * parameters.lambda_v
		for j in range(parameters.m_num_items):
			user_ids = items[j]
			m = len(user_ids)
			if m>0 :
				A = vv + np.dot(parameters.m_U[user_ids,:].T, parameters.m_U[user_ids,:])*a_minus_b
				x = a * np.sum(parameters.m_U[user_ids, :], axis=0) + parameters.lambda_v * (parameters.m_theta[j,:] + parameters.m_gamma[j,:])
				parameters.m_V[j, :] = scipy.linalg.solve(A, x)
			else:
				# m=0, this article has never been rated
				A = np.copy(vv)
				x = parameters.lambda_v * (parameters.m_theta[j,:] + parameters.m_gamma[j,:])
				parameters.m_V[j, :] = scipy.linalg.solve(A, x)
		iteration += 1


def CAttAE(X, data, sparse, data_name, output_name, pretrain):

    T = data["tags"]
    dimension = parameters.dimension
    
    pretraining = pretrain
    train_model, eval_model = AttAE_module(len(X[0]), dimension)
    train_model2, eval_model2 = AttAE_module(len(T[0]), dimension)

    # Pre-Train the two models; or load weights from pretrained models
    if (pretraining):
        print ("pretraining started")
        history = train(train_model, X, X, parameters.mlp_epoch)
        history2 = train(train_model2, T, T, parameters.mlp_epoch)
        print ("pretraining finished")
        if (data_name == 'a'):
            save_model(train_model, 'models/A_model_sparse/mult_nor'+str(dimension))
            save_model(train_model2, 'models/A_model_sparse/tags'+str(dimension))
        elif (data_name == 't'):
            save_model(train_model, 'models/T_model_sparse/mult_nor'+str(dimension))
            save_model(train_model2, 'models/T_model_sparse/tags'+str(dimension))

    else:
        print ("#Loading model weights#")
        if (data_name == 'a'):
            load_model(train_model, 'models/A_model_sparse/mult_nor'+str(dimension))
            load_model(train_model2, 'models/A_model_sparse/tags'+str(dimension))
        elif (data_name == 't'):
            load_model(train_model, 'models/T_model_sparse/mult_nor'+str(dimension))
            load_model(train_model2, 'models/T_model_sparse/tags'+str(dimension))

    parameters.m_theta[:] = get_z_layer(eval_model, X)
    parameters.m_gamma[:] =  get_z_layer(eval_model2, T)
    
    
    counter = 0
    recall, dcg, ndcg  = [np.zeros((3, 7)) for i in range(3)] 
    # Take the average performance of the three different splits
    for data_num in range(1,4):
        print ("===========Split#",data_num,"=============")
        train_user = data["train_users"+str(data_num)]
        train_item = data["train_items"+str(data_num)]
        test_user = data["test_users"+str(data_num)]
        restore()
        #Calculate U, V
        pmf_estimate(train_user, train_item, parameters.n_epochs)
        # Evaluate CATA++
        recall[counter, :], dcg[counter, :], ndcg[counter, :] = evaluate(test_user, 300) 
        counter += 1
        
        if (sparse):
            if (data_name == 'a'):
                scipy.io.savemat('models/A_model_sparse/'+output_name+str(data_num),{"m_U": parameters.m_U, "m_V": parameters.m_V}) 
            elif (data_name == 't'):
                scipy.io.savemat('models/T_model_sparse/'+output_name+str(data_num),{"m_U": parameters.m_U, "m_V": parameters.m_V})
        else:
            if (data_name == 'a'):
                scipy.io.savemat('models/A_model_dense/'+output_name+str(data_num),{"m_U": parameters.m_U, "m_V": parameters.m_V})
            elif (data_name == 't'):
                scipy.io.savemat('models/T_model_dense/'+output_name+str(data_num),{"m_U": parameters.m_U, "m_V": parameters.m_V})
        
        
    print("------------Average Performance-------------") 
    print(" @10    @50    @100   @150   @200   @250   @300") 
    print("-------------------------------------------") 
    print(np.around(np.mean(recall, axis=0),4), " --Recall") 
    print(np.around(np.mean(dcg, axis=0),4), " --DCG") 
    print(np.around(np.mean(ndcg, axis=0),4), " --NDCG") 
    print("-------------------------------------------")  

	
