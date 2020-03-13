import torch
import misc.utils_ppo as utils
from misc.rewards_ppo import get_self_critical_reward


class LossWrapper(torch.nn.Module):
    def __init__(self, model, opt):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = model

        if self.opt.self_critical:
            self.rl_crit = utils.RewardCriterion()
        else:
            self.crit = utils.LanguageModelCriterion()

    def forward(self, fc_feats, att_feats, labels, masks, att_masks, gts, gt_indices):
        opt = self.opt

        out = {}
        if self.opt.self_critical:
            self.model.eval()
            with torch.no_grad():
                greedy_res, _ = self.model(fc_feats, att_feats, att_masks, mode='sample')
            self.model.train()
            gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks,
                                                     opt={'sample_method': opt.train_sample_method,
                                                          'beam_size': opt.train_beam_size,
                                                          'sample_n': opt.train_sample_n},
                                                     mode='sample')
            gts = [gts[_] for _ in gt_indices.tolist()]
            reward = get_self_critical_reward(greedy_res, gts, gen_result, self.opt)
            reward = torch.from_numpy(reward).float().to(gen_result.device)
            loss = self.rl_crit(sample_logprobs, gen_result.data, reward)
            out['reward'] = reward[:, 0].mean()
        else:
            loss = self.crit(self.model(fc_feats, att_feats, labels, att_masks), labels[:, 1:], masks[:, 1:])
        out['loss'] = loss
        return out
