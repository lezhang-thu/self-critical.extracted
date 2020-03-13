import torch
import torch.nn as nn
import torch.nn.functional as F

import misc.utils as utils
from .CaptionModel import CaptionModel


class NewFCModel(CaptionModel):
    def __init__(self, opt):
        super(NewFCModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.seq_length = getattr(opt, 'max_length', 20) or opt.seq_length  # maximum sample length
        self.fc_feat_size = opt.fc_feat_size

        self.fc_embed = nn.Linear(self.fc_feat_size, self.input_encoding_size)
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        self._core = LSTMCore(opt.input_encoding_size, opt.rnn_size)
        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)

        self.vocab = opt.vocab

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(1, bsz, self.rnn_size),
                weight.new_zeros(1, bsz, self.rnn_size))

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):
        batch_size = fc_feats.size(0)
        seq_per_img = seq.shape[0] // batch_size
        state = self.init_hidden(batch_size * seq_per_img)

        outputs = fc_feats.new_zeros(batch_size * seq_per_img, seq.size(1) - 1, self.vocab_size + 1)

        # Prepare the features
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)
        # pp_att_feats is used for attention, we cache it in advance to reduce computation cost

        if seq_per_img > 1:
            p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = utils.repeat_tensors(seq_per_img,
                                                                                      [p_fc_feats, p_att_feats,
                                                                                       pp_att_feats, p_att_masks]
                                                                                      )

        for i in range(seq.size(1) - 1):
            it = seq[:, i].clone()
            # break if all the sequences end
            if i >= 1 and seq[:, i].sum() == 0:
                break

            output, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)
            outputs[:, i] = output

        return outputs

    def get_logprobs_state(self, it, fc_feats, att_feats, p_att_feats, att_masks, state):
        # 'it' contains a word index
        xt = self.embed(it)

        output, state = self.core(xt, fc_feats, att_feats, p_att_feats, state, att_masks)
        logprobs = F.log_softmax(self.logit(output), dim=1)

        return logprobs, state

    def _sample_beam(self, fc_feats, att_feats, att_masks=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        sample_n = opt.get('sample_n', 10)
        # when sample_n == beam_size then each beam is a sample.
        assert sample_n == 1 or sample_n == beam_size, 'when beam search, sample_n == 1 or beam search'
        batch_size = fc_feats.size(0)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        # let's assume this for now
        assert beam_size <= self.vocab_size + 1
        seq = fc_feats.new_zeros((batch_size * sample_n, self.seq_length), dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size * sample_n, self.seq_length, self.vocab_size + 1)
        # let's process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]

        state = self.init_hidden(batch_size)

        # first step, feed bos
        it = fc_feats.new_zeros([batch_size], dtype=torch.long)
        logprobs, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = utils.repeat_tensors(beam_size,
                                                                                  [p_fc_feats, p_att_feats,
                                                                                   pp_att_feats, p_att_masks]
                                                                                  )
        self.done_beams = self.beam_search(state, logprobs, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, opt=opt)
        for k in range(batch_size):
            if sample_n == beam_size:
                for _n in range(sample_n):
                    seq_len = self.done_beams[k][_n]['seq'].shape[0]
                    seq[k * sample_n + _n, :seq_len] = self.done_beams[k][_n]['seq']
                    seqLogprobs[k * sample_n + _n, :seq_len] = self.done_beams[k][_n]['logps']
            else:
                seq_len = self.done_beams[k][0]['seq'].shape[0]
                seq[k, :seq_len] = self.done_beams[k][0]['seq']  # the first beam has highest cumulative score
                seqLogprobs[k, :seq_len] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq, seqLogprobs

    def _sample(self, fc_feats, att_feats, att_masks=None, opt={}):
        sample_method = opt.get('sample_method', 'greedy')
        beam_size = opt.get('beam_size', 1)
        sample_n = int(opt.get('sample_n', 1))
        if beam_size > 1:
            return self._sample_beam(fc_feats, att_feats, att_masks, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size * sample_n)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        if sample_n > 1:
            p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = utils.repeat_tensors(sample_n,
                                                                                      [p_fc_feats, p_att_feats,
                                                                                       pp_att_feats, p_att_masks]
                                                                                      )

        seq = fc_feats.new_zeros((batch_size * sample_n, self.seq_length), dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size * sample_n, self.seq_length, self.vocab_size + 1)
        for t in range(self.seq_length + 1):
            if t == 0:  # input <bos>
                it = fc_feats.new_zeros(batch_size * sample_n, dtype=torch.long)

            logprobs, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)

            # sample the next word
            if t == self.seq_length:  # skip if we achieve maximum length
                break
            it, sampleLogprobs = self.sample_next_word(logprobs, sample_method)

            # stop when all finished
            if t == 0:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)
            it = it * unfinished.type_as(it)
            seq[:, t] = it
            seqLogprobs[:, t] = logprobs
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        return seq, seqLogprobs

    def core(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks):
        # Step 0, feed the input image
        # if (self.training and state[0].is_leaf) or \
        #     (not self.training and state[0].sum() == 0):
        #     _, state = self._core(fc_feats, state)
        # three cases
        # normal mle training
        # Sample
        # beam search (diverse beam search)
        # fixed captioning module.
        is_first_step = (state[0] == 0).all(2).all(0)  # size: B
        if is_first_step.all():
            _, state = self._core(fc_feats, state)
        elif is_first_step.any():
            # This is mostly for diverse beam search I think
            new_state = [torch.zeros_like(_) for _ in state]
            new_state[0][:, ~is_first_step] = state[0][:, ~is_first_step]
            new_state[1][:, ~is_first_step] = state[1][:, ~is_first_step]
            _, state = self._core(fc_feats, state)
            new_state[0][:, is_first_step] = state[0][:, is_first_step]
            new_state[1][:, is_first_step] = state[1][:, is_first_step]
            state = new_state

        return self._core(xt, state)

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        fc_feats = self.fc_embed(fc_feats)

        return fc_feats, att_feats, att_feats, att_masks


class LSTMCore(nn.Module):
    def __init__(self, input_encoding_size, rnn_size):
        super(LSTMCore, self).__init__()
        self.input_encoding_size = input_encoding_size
        self.rnn_size = rnn_size

        # Build a LSTM
        self.i2h = nn.Linear(self.input_encoding_size, 5 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, xt, state):
        all_input_sums = self.i2h(xt) + self.h2h(state[0][-1])
        sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)
        sigmoid_chunk = torch.sigmoid(sigmoid_chunk)
        in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
        out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)

        in_transform = torch.max(
            all_input_sums.narrow(1, 3 * self.rnn_size, self.rnn_size),
            all_input_sums.narrow(1, 4 * self.rnn_size, self.rnn_size))
        next_c = forget_gate * state[1][-1] + in_gate * in_transform
        next_h = out_gate * torch.tanh(next_c)

        output = self.dropout(next_h)
        state = (next_h.unsqueeze(0), next_c.unsqueeze(0))
        return output, state
