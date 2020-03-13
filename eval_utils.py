import json
import os
import sys

import misc.utils_ppo as utils
import numpy as np
import torch
import torch.nn.functional as F

# load coco-caption if available
try:
    sys.path.append("coco-caption")
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap
except:
    print('Warning: coco-caption not available')


def getCOCO(dataset):
    assert 'coco' in dataset
    annFile = os.path.join('coco-caption', 'annotations',
                           'captions_val2014.json')
    return COCO(annFile)


def language_eval(dataset, preds, eval_kwargs, split):
    # create output dictionary
    out = {}

    cache_path = '.cache_{}.json'.format(eval_kwargs['id'])

    coco = getCOCO(dataset)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set
    preds_filt = [p for p in preds if p['image_id'] in valids]
    mean_perplexity = sum([_['perplexity'] for _ in preds_filt]) / len(preds_filt)
    mean_entropy = sum([_['entropy'] for _ in preds_filt]) / len(preds_filt)
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w'))  # serialize to temporary json file. Sigh, COCO API ...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    for metric, score in cocoEval.eval.items():
        out[metric] = score
    # Add mean perplexity
    out['perplexity'] = mean_perplexity
    out['entropy'] = mean_entropy

    imgToEval = cocoEval.imgToEval
    for k in list(imgToEval.values())[0]['SPICE'].keys():
        if k != 'All':
            out['SPICE_' + k] = np.array([v['SPICE'][k]['f'] for v in imgToEval.values()])
            out['SPICE_' + k] = (out['SPICE_' + k][out['SPICE_' + k] == out['SPICE_' + k]]).mean()

    return out


def eval_split(model, loader, eval_kwargs={}):
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    dataset = eval_kwargs.get('dataset', 'coco')

    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)

    n = 0
    predictions = []
    while True:
        data = loader.get_batch(split)
        n = n + len(data['infos'])

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        tmp = [data['fc_feats'],
               data['att_feats'],
               data['att_masks']]
        tmp = [_.cuda() if _ is not None else _ for _ in tmp]
        fc_feats, att_feats, att_masks = tmp
        with torch.no_grad():
            tmp_eval_kwargs = eval_kwargs.copy()
            tmp_eval_kwargs.update({'sample_n': 1})
            seq, seq_logprobs = model(fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode='sample')
            seq = seq.data
            entropy = - (F.softmax(seq_logprobs, dim=2) * seq_logprobs).sum(2).sum(1) / ((seq > 0).float().sum(1) + 1)
            perplexity = - seq_logprobs.gather(2, seq.unsqueeze(2)).squeeze(2).sum(1) / ((seq > 0).float().sum(1) + 1)

        sents = utils.decode_sequence(loader.get_vocab(), seq)

        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent, 'perplexity': perplexity[k].item(),
                     'entropy': entropy[k].item()}
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']
            predictions.append(entry)

        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        else:
            num_images = ix1
        for i in range(n - ix1):
            predictions.pop()

        if 0 <= num_images <= n:
            break

    lang_stats = language_eval(dataset, predictions, eval_kwargs, split)

    # Switch back to training mode
    model.train()
    return None, None, lang_stats
