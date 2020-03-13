
# Self-critical Sequence Training for Image Captioning

This repo is an extracted version of [ruotianluo / self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch/tree/master). 
The only thing done is to remove so many hyper-parameters of the original repo, so
one can see clearly how self-critical is running. Also only `NewFCModel` is remained here.
Only for research purposes.

## Prepare data

### Download COCO captions and preprocess them

Download preprocessed coco captions from [link](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) from Karpathy's homepage. Extract `dataset_coco.json` from the zip file and copy it in to `data/`. This file provides preprocessed captions and also standard train-val-test splits.

Then do:

```bash
$ python scripts/prepro_labels.py --input_json data/dataset_coco.json --output_json data/cocotalk.json --output_h5 data/cocotalk
```

`prepro_labels.py` will map all words that occur less than 5 times to a special `UNK` token, and create a vocabulary for all the remaining words. The image information and vocabulary are dumped into `data/cocotalk.json` and discretized caption data are dumped into `data/cocotalk_label.h5`.

### Download Bottom-up features

#### Convert from peteanderson80's original file
Download pre-extracted features from [link](https://github.com/peteanderson80/bottom-up-attention). Choose the adaptive one.

For example:
```
mkdir data/bu_data; cd data/bu_data
wget https://imagecaption.blob.core.windows.net/imagecaption/trainval.zip
unzip trainval.zip
```

Then:

```bash
$ python script/make_bu_data.py --output_dir data/cocobu
```
Note: Use Python 2, instead of Python 3. This is due to `make_bu_data.py`.

This will create `data/cocobu_fc`, `data/cocobu_att` and `data/cocobu_box`.

### Prepare for self-critical

When traing using self-critical, you should first preprocess the dataset 
and get the cache for calculating cider score.

```bash
$ python scripts/prepro_ngrams.py --input_json data/dataset_coco.json --dict_json data/cocotalk.json --output_pkl data/coco-train --split train
```

## Get cider and coco-caption

Get [cider](https://github.com/ruotianluo/cider) and [coco-caption](https://github.com/ruotianluo/coco-caption).

## Train using XE first

```bash
$ python train.py --id fc --cfg configs/fc.yml
```
The checkpoints are dumped into the folder `log_$id`. By default
only the best-performing checkpoint on validation and the latest checkpoint are saved.
The training is running for `max_epoch` epochs specified in `configs/fc.yml`.

## Train using self-critical thereafter

After the XE training, we can restart the training using self-critical for an extra number
of epochs, which is the difference of `max_epcoh`s in `configs/fc_rl.yml` and `configs/fc.yml`.

We need to copy the pretrained XE trained model, as follows.

```bash
mkdir log_fc_rl
cp log_fc/model.pth log_fc_rl
cp log_fc/infos_fc.pkl log_fc_rl/infos_fc_rl.pkl
```

With all these, the training can be started.

```bash
$ python train.py --id fc_rl --cfg configs/fc_rl.yml
```

Note: Watch the prompts. Make sure you see two messages `infos load success` and `model load success`.
A common pitfall is forgetting the `--id fc_rl` when starting training.

## Watch the training

This are literally no prompts when training, except the messages given by `coco-caption`.
The training process can be watched by `$ tensorboard --logdir=log_fc` for XE or `$ tensorboard --logdir=log_fc_rl` for self-critical, and
visit the prompted ***IP*** address and port (given by `tensorboard`) in browser.

## How to evaluate

To evaluate self-critical, run 

```bash
$ python eval.py --cfg configs/eval-fc_rl.yml
```

Or manually edit `configs/eval-fc_rl.yml` to evaluate XE result.

## Results

As said in the original repo,

> The scores are just used to verify if you are getting things right. 
> If the scores you get is close to the number I give (it could be higher or lower), then it's ok.

| Name                        									  | CIDEr     | SPICE    |
| :---:                     									    | :---:     | :---:    |
| [NewFCModel](configs/fc.yml)                    | 1.007     | 0.188    |
| [NewFCModel + self-critical](configs/fc_rl.yml) | 1.090     | 0.194    |

