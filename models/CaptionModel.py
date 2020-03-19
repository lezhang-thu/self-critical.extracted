import torch
import torch.nn as nn

import misc.utils as utils


class CaptionModel(nn.Module):
    def __init__(self):
        super(CaptionModel, self).__init__()

    def forward(self, *args, **kwargs):
        mode = kwargs.get('mode', 'forward')
        if 'mode' in kwargs:
            del kwargs['mode']
        return getattr(self, '_' + mode)(*args, **kwargs)

    # implements beam search
    # calls beam_step and returns the final set of beams
    def beam_search(self, init_state, init_logprobs, *args, **kwargs):
        # does one step of classical beam search
        def beam_step(logprobs, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state):
            # INPUTS:
            # t                 : time step
            # beam_seq          : tensor containing the beams
            # beam_seq_logprobs : tensor containing the beam logprobs
            # beam_logprobs_sum : tensor containing joint logprobs
            # OUTPUTS:
            # beam_seq          : tensor containing the word indices of the decoded captions N x b x (t + 1)
            # beam_seq_logprobs : log-probability of each decision made, N x b x (t + 1) x V
            # beam_logprobs_sum : joint log-probability of each beam N x b
            batch_size = beam_logprobs_sum.shape[0]
            vocab_size = logprobs.shape[-1]
            logprobs = logprobs.reshape(batch_size, -1, vocab_size)  # N x b x V

            candidate_logprobs = beam_logprobs_sum.unsqueeze(
                -1) + logprobs  # beam_logprobs_sum N x b, logprobs is N x b x V
            ys, ix = torch.sort(candidate_logprobs.reshape(candidate_logprobs.shape[0], -1), -1, True)
            ys, ix = ys[:, :beam_size], ix[:, :beam_size]
            beam_ix = ix // vocab_size  # N x b which beam
            selected_ix = ix % vocab_size  # N x b which word
            state_ix = (beam_ix + torch.arange(batch_size).type_as(beam_ix).unsqueeze(-1) * logprobs.shape[1]).reshape(
                -1)  # N * b which in N x b beams
            # gather according to beam_ix
            if t > 0:
                beam_seq = beam_seq.gather(1, beam_ix.unsqueeze(-1).expand_as(beam_seq))
                beam_seq_logprobs = beam_seq_logprobs.gather(1, beam_ix.unsqueeze(-1).unsqueeze(-1).expand_as(
                    beam_seq_logprobs))

            beam_seq = torch.cat([beam_seq, selected_ix.unsqueeze(-1)], -1)  # beam_seq N x b x (t + 1)
            beam_logprobs_sum = beam_logprobs_sum.gather(1, beam_ix) + \
                                logprobs.reshape(batch_size, -1).gather(1, ix)
            beam_ix = beam_ix.unsqueeze(-1).expand(-1, -1, vocab_size)

            beam_logprobs = logprobs.reshape(batch_size, -1, vocab_size).gather(1, beam_ix)  # N x b x V
            beam_seq_logprobs = torch.cat([
                beam_seq_logprobs,
                beam_logprobs.reshape(batch_size, -1, 1, vocab_size)], 2)

            new_state = [None for _ in state]
            for _ix in range(len(new_state)):
                # copy over state in previous beam q to new beam at vix
                new_state[_ix] = state[_ix][:, state_ix]
            state = new_state
            return beam_seq, beam_seq_logprobs, beam_logprobs_sum, state

        opt = kwargs['opt']
        beam_size = opt.get('beam_size', 10)
        length_penalty = utils.penalty_builder(opt.get('length_penalty', ''))

        batch_size = init_logprobs.shape[0]
        device = init_logprobs.device
        # INITIALIZATIONS
        beam_seq = torch.LongTensor(batch_size, beam_size, 0).to(device)
        beam_seq_logprobs = torch.FloatTensor(batch_size, beam_size, 0, self.vocab_size + 1).to(device)
        beam_logprobs_sum = torch.zeros(batch_size, 1).to(device)

        # logprobs predicted in last time step, shape (beam_size, vocab_size + 1)
        done_beams = [[] for _ in range(batch_size)]
        state = [_.clone() for _ in init_state]
        logprobs = init_logprobs
        # END INIT

        # Chunk elements in the args
        args = list(args)

        for t in range(self.seq_length):
            # infer new beams
            beam_seq, \
            beam_seq_logprobs, \
            beam_logprobs_sum, \
            state = beam_step(logprobs,
                              beam_size,
                              t,
                              beam_seq,
                              beam_seq_logprobs,
                              beam_logprobs_sum,
                              state)

            # if time's up ... or if end token is reached then copy beams
            for k in range(batch_size):
                is_end = beam_seq[k, :, t] == 0
                if t == self.seq_length - 1:
                    is_end.fill_(1)
                for vix in range(beam_size):
                    if is_end[vix]:
                        final_beam = {
                            'seq': beam_seq[k, vix].clone(),
                            'logps': beam_seq_logprobs[k, vix].clone(),
                            'p': beam_logprobs_sum[k, vix].item()
                        }
                        final_beam['p'] = length_penalty(t + 1, final_beam['p'])
                        done_beams[k].append(final_beam)
                beam_logprobs_sum[k, is_end] -= 1000

            # one step forward
            it = beam_seq[:, :, t].reshape(-1).detach()
            logprobs, state = self.get_logprobs_state(it, *(
                    args + [state]))

        # all beams are sorted by their log-probabilities
        done_beams = [sorted(done_beams[k], key=lambda x: - x['p'])[:beam_size]
                      for k in range(batch_size)]
        return done_beams

    def sample_next_word(self, logprobs, sample_method):
        assert sample_method in {'greedy', 'sample'}
        if sample_method == 'greedy':
            sampleLogprobs, it = torch.max(logprobs, 1)
            it = it.view(-1).long()
        else:
            it = torch.distributions.Categorical(logits=logprobs).sample()
            sampleLogprobs = logprobs.gather(1, it.unsqueeze(1))  # gather the logprobs at sampled positions
        return it.detach(), sampleLogprobs

    def decode_sequence(self, seq):
        return utils.decode_sequence(self.vocab, seq)
