# About The Project
Implementation of paper *Lightweight Approaches to DNN Regression Error Reduction: An Uncertainty Alignment Perspective*

Note: the reported NFR (negative flip rate) is essentially the regression error rate.  


## Requirements

- python >= 3.6
- pytorch >= 1.4



## Repository contents

| file               | description                                |
| ------------------ | ------------------------------------------ |
| data/              | The datasets dir                           |
| save/              | The results dir                            |
| train_model.py     | The training code                          |
| test_model.py      | The test code to compute Acc and Reg       |
| regbug_kd.py       | The LM and KD methods in the paper         |
| regbug_lsa.py      | The LSA (supervise adequacy) method        |
| regbug_bayesian.py | The MB, MBME, and CR methods in the paper  |
| regbug_reducing.py | The avg, max, and our methods in the paper |



## Datasets and models

- All supported datasets and models are in  `./dataset` and `./models`, respectively. 

- To train the model, use the following command. 

  > python train_model.py --dataset mnist --model mnistlenet --learning_algorithm 'adam' --learning_rate 5e-4 --epochs 30 --dataset_size 1.0 --trial 1 

- You can change the *dataset* and *model* parameters to derive other results. 

- To reproduce the results in the paper, see `run_training.sh`

## Regression errors reduction without training

- To conduct the regression error reduction by Avg, run the following command.

  > python regbug_reducing.py --dataset emnist --dataset_size 1.0 --ensemble 'average' \
  >
  >    --old_model mnistlenet --oldmodel_path ./save/models/mnistlenet_mnist_1.0_lr_0.0005_decay_0.0_trial_1/mnistlenet_dataset_1.0_best.pth \
  >
  >    --new_model mnistlenet --newmodel_path ./save/models/mnistlenet_emnist_0.1_lr_0.0005_decay_0.0_trial_1/mnistlenet_dataset_0.1_best.pth

-  You can change the *ensemble* parameter to conduct other methods, including
  
    > *average, maximum, deepgini, dropout, perturb, scaling_unlabel*. 
  
    Note that *scaling unlabel* is the TS method in the paper.

- To reproduce the results in the paper, see `run_debugging.sh`.



## Regression errors reduction with training

- To conduct the regression error reduction by LM and KD, run the following command. 

  > python regbug_kd.py --dataset stl10 --distill 'lm' --dataset_size 1.0 -r 1.0 -a 0.0001 --kd_T 100.0 \
  >
  >    --old_model mnistlenet --oldmodel_path ./save/models/mnistlenet_mnist_1.0_lr_0.0005_decay_0.0_trial_1/mnistlenet_dataset_1.0_best.pth \
  >
  >    --new_model mnistlenet --newmodel_path ./save/models/mnistlenet_emnist_0.1_lr_0.0005_decay_0.0_trial_1/mnistlenet_dataset_0.1_best.pth

- To reproduce the results in the paper, see `run_kd.sh`.



## Regression errors reduction with Bayesian method

- To conduct the regression error reduction by MB, MBME, and CR, run the following command. 

  > python regbug_bayesian.py --dataset emnist --dataset_size 1.0 --combine 'CostRatio' \
  >
  >    --old_model mnistlenet --oldmodel_path ./save/models/mnistlenet_mnist_1.0_lr_0.0005_decay_0.0_trial_1/mnistlenet_dataset_1.0_best.pth \
  >
  >    --new_model mnistlenet --newmodel_path ./save/models/mnistlenet_emnist_0.1_lr_0.0005_decay_0.0_trial_1/mnistlenet_dataset_0.1_best.pth

- To reproduce the results in the paper, see `run_bayes.sh`.
