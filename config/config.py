import argparse
import yaml
from collections import namedtuple

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default=None)
parser.add_argument('--batch_size', default=None)
parser.add_argument('--max_epoch', default=None)
parser.add_argument('--shuffle', default=None)
parser.add_argument('--learning_rate', default=None)
parser.add_argument('--train_path', default=None)
parser.add_argument('--dev_path', default=None)
parser.add_argument('--test_path', default=None)
parser.add_argument('--predict_path', default=None)
parser.add_argument('--weight_decay', default=None)
parser.add_argument('--warm_up_ratio', default=None)
parser.add_argument('--loss_func', default=None)
parser.add_argument('--run_name', default=None)
parser.add_argument('--project_name', default=None)
parser.add_argument('--entity', default=None) # wandb team name
args = parser.parse_args()

with open("./config/config.yaml") as f:
    config_dict = yaml.load(f, Loader=yaml.FullLoader)

for key in args.__dict__:
    if args.__dict__[key] is not None:
        config_dict[key] = args.__dict__[key]

if isinstance(config_dict["learning_rate"], str):
    config_dict["learning_rate"] = float(config_dict["learning_rate"])
Config = namedtuple("config", config_dict.keys())
config = Config(**config_dict)

if __name__ == "__main__":
    print(config._asdict())