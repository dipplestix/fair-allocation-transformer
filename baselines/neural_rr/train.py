import torch.optim as optim
from data import SyntheticDataset, create_loader
from utils import (set_seed, seed_worker, save_model,
                   select_loss_functions, select_valuations, select_allocator)
from argparse import ArgumentParser, Namespace
import datetime
from EEF1NN import EEF1NN
from neural_rr import NeuralRR
from training_utils import train_model
import json
import os

def main(args:Namespace):
    # set seed
    set_seed(args.seed)

    # Load hyper-parameters
    with open(args.hyperparameter_path, 'r') as f:
        hp_configs = json.load(f)

    model_config = hp_configs['model']
    loss_config = hp_configs['loss']
    data_config = hp_configs['data']
    optimizer_config = hp_configs['optimizer']

    # create dataloader
    generator_kwargs = {
        'low': getattr(args, 'low', None),
        'high': getattr(args, 'high', None),
        'eps': getattr(args, 'eps', None)
    }
    generator_kwargs = dict((key, val) for (key,val) in generator_kwargs.items() if val is not None)
    generator = select_valuations(**generator_kwargs, choice=args.generator) if hasattr(args, 'generator') else None
    allocator = select_allocator(choice=args.allocator) if hasattr(args, 'allocator') else None
    dataset = SyntheticDataset(num_agents=args.num_agents,
                               num_objects=args.num_objects,
                               num_samples=args.num_samples,
                               generator=generator,
                               allocator=allocator,
                               saved_path=getattr(args, 'train_data_path', None))
    train_loader = create_loader(dataset=dataset,
                                 batch_size=data_config['batch_size'], 
                                 pin_memory=(args.device=='cuda'),
                                 num_workers=2,
                                 worker_init_fn=seed_worker,
                                 shuffle=True)

    # model, loss, optimizer
    model_class = EEF1NN if args.model_name == 'EEF1NN' else NeuralRR
    model = model_class(**model_config)
    print(model)
    criterion = select_loss_functions(loss_config['name'], **loss_config['params'])
    optimizer = optim.Adam(model.parameters(), lr=optimizer_config['lr'])

    # train
    train_loss = train_model(model=model, 
                             train_loader=train_loader, 
                             criterion=criterion, 
                             optimizer=optimizer, 
                             device=args.device, 
                             epochs=data_config['epochs'])
    print(f"{train_loss=}")

    # Save parameters and checkpoint
    date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    checkpoint_name = args.model_name + "_" + date + ".pt"
    os.makedirs(args.save_dir, exist_ok=True)
    setting_name = os.path.join(args.save_dir, 'train_params_' + date + '.json')
    args_dict = vars(args)
    args_dict["hp"] = hp_configs

    with open(setting_name, 'w') as f:
        json.dump(args_dict, f, indent=4)

    save_model(model=model, directory=args.save_dir, filename=checkpoint_name)

if __name__ == '__main__':
    parser = ArgumentParser()

    # required arguments
    parser.add_argument('--hyperparameter_path', type=str, required=True, help="Path to hyper-parameters.")

    # optinal arguments
    parser.add_argument('--allocator', choices=['MUW'], default='MUW', help='Target data. (default: MUW)')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', help="Device. ('cuda' or 'cpu', default: 'cpu')")
    parser.add_argument('--generator', choices=['AverageNoise'], default='AverageNoise', help="Valuation Generator. (default: 'AverageNoise')")
    parser.add_argument('--low', type=float, default=1.0, help="Valuation lower bound. (default: 1.0)")
    parser.add_argument('--high', type=float, default=2.0, help="Valuation upper bound. (default: 2.0)")
    parser.add_argument('--model_name', choices=['EEF1NN', 'NRR'], default='NRR', help="Model to be trained. (default: NRR)")
    parser.add_argument('--num_agents', type=int, default=15, help="Number of agents. (default: 15)")
    parser.add_argument('--num_objects', type=int, default=5, help="Number of objects. (default: 10)")
    parser.add_argument('--num_samples', type=int, default=100, help="Number of samples. (default: 100)")
    parser.add_argument('--save_dir', type=str, default="train_results/", help="Directory to save checkpoints. (default: ./train_results/)")
    parser.add_argument('--seed', type=int, default=42, help="Seed. (default: 42)")
    parser.add_argument('--train_data_path', required=False, help="path to train data.")
    main(parser.parse_args())
