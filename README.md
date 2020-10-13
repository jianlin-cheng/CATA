

# Collaborative Dual Attentive Autoencoder for Recommending Scientific Articles

## Overview
> This is the implementation of our two recent models for recommending scientific articles. The first model, CATA, utilizes articles’ titles and abstracts via a single attentive autoencoder, while the second model, CATA++, utilizes additional articles' information (such as tags and citations between articles) via two parallel attentive autoencoders. 

## Papers
1. Collaborative Attentive Autoencoder for Scientific Article Recommendation (<a href="https://ieeexplore.ieee.org/document/8999062" target="_blank">_**ICMLA 2019**_</a>)
2. CATA++: A Collaborative Dual Attentive Autoencoder Method for Recommending Scientific Articles (<a href="https://ieeexplore.ieee.org/document/9217428" target="_blank">_**Access 2020**_</a>)

## Datasets

  | Dataset	  | #users	| #articles | #pairs | sparsity 	 |
  | --------	 | ---		| ---	 | --- 		  | ---	 |
  | Citeulike-a| 5,551	| 16,980	 | 204,986  | 99.78% |
  | Citeulike-t| 7,947	| 25,975	 | 134,860	| 99.93% |
  
## Requirements
- Python 3
- Tensorflow
- Keras

## How to Run

### Configurations
You can evaluate our models with different settings in terms of:
1. data_name `-d`: 'a' for citeulike-a and 't' for citeulike-t. [*Optional*] [*Default=citeulike-a*]
2. sparse `-s`: '0', 'no', 'false', or 'f' for dense. [*Optional*] [*Default=sparse*]
3. pretrain `-pretrain`: '1', 'yes', 'true' or 't' for pretrain the attentive autoencoder. [*Optional*] [*Default=No-pretraining*]
4. epochs `-e`: Number of epochs to pretrain the attentive autoencoder. [*Optional*] [*Default=150*]
5. lambda_u `-u`: value of lambda_u. [*Optional*] [*Default=10*] 
6. lambda_v `-v`: value of lambda_v. [*Optional*] [*Default=0.1*]
7. pmf_epochs `-pe`: Number of iterations for PMF. [*Optional*] [*Default=100*]
8. output_name `-o` : Name of the output file. [*Optional*]
9. latent_size `-l` :Size of latent space. [*Optional*] [*Default=50*]

### Examples
- To run CATA model, use test_CATA.py file.
- To run CATA++ model, use test_CATA++.py file.

- **Example 1**: Run CATA++ model for citeulike-t dataset with the dense setting. Also, always pretrain the two autoencoders if you run the code for the first time by setting `-pretrain 1`:
	- `python3 test_CATA++.py -pretrain 1 -d 't' -s 0`
- **Example 2**: Run CATA++ model for citeulike-t dataset with the sparse setting:
	- `python3 test_CATA++.py -pretrain 0 -d 't' -s 1`
- **Example 3**: Run CATA model for citeulike-a dataset with the sparse setting:
	- `python3 test_CATA.py -pretrain 0 -d 'a' -s 1`

