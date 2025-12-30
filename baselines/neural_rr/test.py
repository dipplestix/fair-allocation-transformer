import torch
from data import SyntheticDataset, create_loader
from utils import (set_seed, select_allocator, select_metrics, select_valuations)
from argparse import ArgumentParser, Namespace
from EEF1NN import EEF1NN
from neural_rr import NeuralRR
from baselines import RR
from training_utils import test_model
import json
import numpy as np
import os
import datetime
import pandas as pd

def main(args:Namespace):
    # set seed
    set_seed(args.seed)

    # Load hyper-parameters
    if args.model_name in {'EEF1NN', 'NRR'}:
        with open(args.hyperparameter_path, 'r') as f:
            hp_configs = json.load(f)

        model_config = hp_configs['model']
    else:
        hp_configs = dict()
        model_config = dict()

    # create dataloader
    generator_kwargs = {
        'low': getattr(args, 'low', None),
        'high': getattr(args, 'high', None),
        'eps': getattr(args, 'eps', None)
    }
    generator_kwargs = dict((key, val) for (key,val) in generator_kwargs.items() if val is not None)
    generator = select_valuations(**generator_kwargs, choice=args.generator)
    allocator = select_allocator(choice=args.allocator)
    dataset = SyntheticDataset(num_agents=args.num_agents,
                               num_objects=args.num_objects,
                               num_samples=args.num_samples,
                               generator=generator,
                               allocator=allocator,
                               saved_path=getattr(args, 'test_data_path', None))
    test_loader = create_loader(dataset=dataset, 
                                batch_size=128, 
                                pin_memory=(args.device=='cuda'),
                                num_workers=2,
                                shuffle=False)

    # model
    if args.model_name in {'EEF1NN', 'NRR'}:
        model_class = EEF1NN if args.model_name == 'EEF1NN' else NeuralRR
        model = model_class(**model_config)
        model.load_state_dict(torch.load(args.model_path))
    elif args.model_name == 'RR':
        model = RR()
    
    # metric
    test_metrics = dict()
    print(f"Test metrics: {args.metric}")
    for choice in args.metric:
        metric = select_metrics(choice=choice)

        # test
        test_metric = test_model(model=model,
                                 test_loader=test_loader,
                                 metric=metric,
                                 device=args.device)
        test_metrics[repr(metric)] = test_metric

        mean, std = np.mean(test_metric), np.std(test_metric)
        print(metric, f": {mean:.5f} ±{std:.5f}")

    if args.save:
        os.makedirs('test_results', exist_ok=True)
        date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        json_filename = os.path.join('test_results', f'{date}.json')
        csv_filename = os.path.join('test_results', f'{date}.csv')

        with open(json_filename, 'w') as f:
            args_dict = vars(args)
            args_dict['hp'] = hp_configs
            json.dump(args_dict, f, indent=4)

        df = pd.DataFrame(test_metrics)
        df['model'] = getattr(args, 'csv_model_name', args.model_name)
        df['seed'] = args.seed
        df.to_csv(csv_filename, index=False)

if __name__ == '__main__':
    parser = ArgumentParser()

    # required arguments
    parser.add_argument('--hyperparameter_path', type=str, required=False, help="Path to hyper-parameters.")
    parser.add_argument('--model_path', type=str, required=False, help="Path to trained model.")

    # optinal arguments
    parser.add_argument('--allocator', choices=['MUW'], default='MUW', help='Target data. (default: MUW)')
    parser.add_argument('--csv_model_name', type=str, required=False, help='Model name in csv. (default: args.model_name)')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', help="Device. ('cuda' or 'cpu', default: 'cpu')")
    parser.add_argument('--generator', choices=['AverageNoise'], default='AverageNoise', help="Valuation Generator. (default: 'AverageNoise')")
    parser.add_argument('--low', type=float, default=1.0, help="Valuation lower bound. (default: 1.0)")
    parser.add_argument('--high', type=float, default=2.0, help="Valuation upper bound. (default: 2.0)")
    parser.add_argument('--metric', nargs="+", choices=['Hamm', 'SCW', 'rSCW', 'EF1Hard'], default='rSCW', help="Metric. (default: 'rSCW')")
    parser.add_argument('--model_name', choices=['EEF1NN', 'NRR', 'RR'], default='NRR', help="Model to be tested. (default: NRR)")
    parser.add_argument('--num_agents', type=int, default=10, help="Number of agents. (default: 15)")
    parser.add_argument('--num_objects', type=int, default=20, help="Number of objects. (default: 5)")
    parser.add_argument('--num_samples', type=int, default=100, help="Number of objects. (default: 100)")
    parser.add_argument('--save', action='store_true', help='Whether to save results into test_results/.')
    parser.add_argument('--seed', type=int, default=42, help="Seed. (default: 42)")
    parser.add_argument('--test_data_path', type=str, required=False, help="path to test data.")

    args = parser.parse_args()

    if args.model_name in {'EEF1NN', 'NRR'}:
        assert args.hyperparameter_path is not None, f"Required '--hyperparameter_path' for model {args.model_name}"
        assert args.model_path is not None, f"Required '--model_path' for model {args.model_name}"

    main(args)
