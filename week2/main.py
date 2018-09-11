from __future__ import print_function
import random
import torch
import torch.backends.cudnn as cudnn

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from config import get_config
from train import Trainer

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(config):
    if config.save_dir is None:
        config.save_dir = 'samples'
    os.system('mkdir {0}'.format(config.save_dir))

    config.manual_seed = random.randint(1, 10000)
    print("Random Seed: ", config.manual_seed)

    random.seed(config.manual_seed)
    torch.manual_seed(config.manual_seed)
    torch.cuda.manual_seed_all(config.manual_seed)

    cudnn.benchmark = True

    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    config = get_config()
    main(config)