import argparse
import sys

import misc.utils_ppo as utils

import eval_utils
import models
from dataloader import *
from misc.config import CfgNode

# load coco-caption if available
try:
    sys.path.append("coco-caption")
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap
except:
    print('Warning: coco-caption not available')

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default=None,
                    help='configuration')

args = parser.parse_args()

cn = CfgNode.load_yaml_with_base(args.cfg)
for k, v in cn.items():
    setattr(args, k, v)
opt = parser.parse_args(namespace=args)

with open(opt.infos_path, 'rb') as f:
    infos = utils.pickle_load(f)

for k in vars(infos['opt']).keys():
    if k not in vars(opt):
        vars(opt).update({k: vars(infos['opt'])[k]})  # copy over options from model

print(sorted(dict(set(vars(opt).items())).items(), key=lambda x: x[0]))

vocab = infos['vocab']  # ix -> word mapping

# Setup the model
opt.vocab = vocab
model = models.setup(opt)
del opt.vocab
model.load_state_dict(torch.load(opt.model))
model.cuda()

loader = DataLoader(opt)

# When eval using provided pre-trained model, the vocab may be different from what you have in your cocotalk.json
# So make sure to use the vocab in infos file.
loader.dataset.ix_to_word = infos['vocab']
# Set sample options
opt.dataset = opt.input_json

_, _, lang_stats = eval_utils.eval_split(model, loader, vars(opt))
print(lang_stats)
