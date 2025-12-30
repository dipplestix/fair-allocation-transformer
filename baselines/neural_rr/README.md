# Learning Fair and Preferable Allocations through Neural Network
This is the implementation of the paper.

## Installation
* Install the required packages:
```bash
pip install -r requirements.txt
```

## Train
* Example script for training neural RR (to train EEF1NN, change `hp_path` and `model_name` accordingly):

```bash
hp_path="hyper_parameter/hp_nrr.json"
model_name="NRR"

python3 train.py \
    --hyperparameter_path $hp_path\
    --model_name $model_name\
    --num_samples <number_of_samples>\
    --num_agents <number_of_agents>\
    --num_objects <number_of_goods>\
    --seed <train_seed>\
    --low <lower_bound_of_mean_valuation>\
    --high <upper_bound_of_mean_valuation>\
    --allocator "MUW"
```

## Test
* Example script for testing neural RR (to test EEF1NN, change `hp_path` and `model_name` accordingly):

```bash
hp_path="hyper_parameter/hp_nrr.json"
model_name="NRR"

python3 test.py \
    --hyperparameter_path $hp_path \
    --model_path <path_to_trained_model>\
    --model_name $model_name \
    --num_samples <number_of_samples> \
    --num_agents <number_of_agents> \
    --num_objects <number_of_goods> \
    --seed <test_seed> \
    --low <lower_bound_of_mean_valuation>\
    --high <upper_bound_of_mean_valuation>\
    --allocator "MUW"\
    --metric Hamm rSCW EF1Hard\
    --device <cpu_or_cuda>\
    --save
```