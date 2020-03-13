import torch
import torch.nn as nn
import torch.nn.functional as F
import misc.utils_ppo as utils


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
        def beam_step(logprobs, unaug_logprobs, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state):
            # INPUTS:
            # logprobs : probabilities augmented after diversity (N * b) x V
            # beam_size: obvious
            # t        : time instant
            # beam_seq         : tensor containing the beams
            # beam_seq_logprobs: tensor containing the beam logprobs
            # beam_logprobs_sum: tensor containing joint logprobs
            # OUTPUTS:
            # beam_seq          : tensor containing the word indices of the decoded captions N x b x l
            # beam_seq_logprobs : log-probability of each decision made, N x b x l x V
            # beam_logprobs_sum : joint log-probability of each beam N x b

            batch_size = beam_logprobs_sum.shape[0]
            vocab_size = logprobs.shape[-1]
            logprobs = logprobs.reshape(batch_size, -1, vocab_size)  # N x b x V
            if t == 0:
                assert logprobs.shape[1] == 1
                beam_logprobs_sum = beam_logprobs_sum[:, :1]
            candidate_logprobs = beam_logprobs_sum.unsqueeze(
                -1) + logprobs  # beam_logprobs_sum N x b, logprobs is N x b x V
            ys, ix = torch.sort(candidate_logprobs.reshape(candidate_logprobs.shape[0], -1), -1, True)
            ys, ix = ys[:, :beam_size], ix[:, :beam_size]
            beam_ix = ix // vocab_size  # N x b which beam
            selected_ix = ix % vocab_size  # N x b which word
            state_ix = (beam_ix + torch.arange(batch_size).type_as(beam_ix).unsqueeze(-1) * logprobs.shape[1]).reshape(
                -1)  # N * b which in N x b beams

            if t > 0:
                # gather according to beam_ix
                assert (beam_seq.gather(1, beam_ix.unsqueeze(-1).expand_as(beam_seq)) ==
                        beam_seq.reshape(-1, beam_seq.shape[-1])[state_ix].view_as(beam_seq)).all()
                beam_seq = beam_seq.gather(1, beam_ix.unsqueeze(-1).expand_as(beam_seq))

                beam_seq_logprobs = beam_seq_logprobs.gather(1, beam_ix.unsqueeze(-1).unsqueeze(-1).expand_as(
                    beam_seq_logprobs))

            beam_seq = torch.cat([beam_seq, selected_ix.unsqueeze(-1)], -1)  # beam_seq N x b x l
            beam_logprobs_sum = beam_logprobs_sum.gather(1, beam_ix) + \
                                logprobs.reshape(batch_size, -1).gather(1, ix)
            assert (beam_logprobs_sum == ys).all()
            _tmp_beam_logprobs = unaug_logprobs[state_ix].reshape(batch_size, -1, vocab_size)
            beam_logprobs = unaug_logprobs.reshape(batch_size, -1, vocab_size).gather(1,
                                                                                      beam_ix.unsqueeze(-1).expand(-1,
                                                                                                                   -1,
                                                                                                                   vocab_size))  # N x b x V
            assert (_tmp_beam_logprobs == beam_logprobs).all()
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
        group_size = opt.get('group_size', 1)
        length_penalty = utils.penalty_builder(opt.get('length_penalty', ''))
        bdash = beam_size // group_size  # beam per group

        batch_size = init_logprobs.shape[0]
        device = init_logprobs.device
        # INITIALIZATIONS
        beam_seq_table = [torch.LongTensor(batch_size, bdash, 0).to(device) for _ in range(group_size)]
        beam_seq_logprobs_table = [torch.FloatTensor(batch_size, bdash, 0, self.vocab_size + 1).to(device) for _ in
                                   range(group_size)]
        beam_logprobs_sum_table = [torch.zeros(batch_size, bdash).to(device) for _ in range(group_size)]

        # logprobs predicted in last time step, shape (beam_size, vocab_size + 1)
        done_beams_table = [[[] for __ in range(group_size)] for _ in range(batch_size)]
        state_table = [[_.clone() for _ in init_state] for _ in range(group_size)]
        logprobs_table = [init_logprobs.clone() for _ in range(group_size)]
        # END INIT

        # Chunk elements in the args
        args = list(args)
        args = utils.split_tensors(group_size, args)  # For each arg, turn (Bbg)x... to (Bb)x(g)x...
        args = [[args[i][j] for i in range(len(args))] for j in range(group_size)]

        for t in range(self.seq_length):
            # add diversity
            logprobs = logprobs_table[0]

            # diversity is added here
            # the function directly modifies the logprobs values and hence, we need to return
            # the unaugmented ones for sorting the candidates in the end.
            # for historical reasons :-)
            unaug_logprobs = logprobs.clone()

            # infer new beams
            beam_seq_table[0], \
            beam_seq_logprobs_table[0], \
            beam_logprobs_sum_table[0], \
            state_table[0] = beam_step(logprobs,
                                       unaug_logprobs,
                                       bdash,
                                       t,
                                       beam_seq_table[0],
                                       beam_seq_logprobs_table[0],
                                       beam_logprobs_sum_table[0],
                                       state_table[0])

            # if time's up ... or if end token is reached then copy beams
            for b in range(batch_size):
                is_end = beam_seq_table[0][b, :, t] == 0
                assert beam_seq_table[0].shape[-1] == t + 1
                if t == self.seq_length - 1:
                    is_end.fill_(1)
                for vix in range(bdash):
                    if is_end[vix]:
                        final_beam = {
                            'seq': beam_seq_table[0][b, vix].clone(),
                            'logps': beam_seq_logprobs_table[0][b, vix].clone(),
                            'unaug_p': beam_seq_logprobs_table[0][b, vix].sum().item(),
                            'p': beam_logprobs_sum_table[0][b, vix].item()
                        }
                        final_beam['p'] = length_penalty(t + 1, final_beam['p'])
                        done_beams_table[b][0].append(final_beam)
                beam_logprobs_sum_table[0][b, is_end] -= 1000

            # move the current group one step forward in time

            it = beam_seq_table[0][:, :, t].reshape(-1)
            logprobs_table[0], state_table[0] = self.get_logprobs_state(it.cuda(), *(
                    args[0] + [state_table[0]]))
            logprobs_table[0] = F.log_softmax(logprobs_table[0], dim=-1)

        # all beams are sorted by their log-probabilities
        done_beams_table = [[sorted(done_beams_table[b][i], key=lambda x: -x['p'])[:bdash] for i in range(group_size)]
                            for b in range(batch_size)]
        done_beams = [sum(_, []) for _ in done_beams_table]
        return done_beams

    def sample_next_word(self, logprobs, sample_method):
        assert sample_method in {'greedy', 'sample'}
        if sample_method == 'greedy':
            sampleLogprobs, it = torch.max(logprobs.data, 1)
            it = it.view(-1).long()
        else:
            it = torch.distributions.Categorical(logits=logprobs.detach()).sample()
            sampleLogprobs = logprobs.gather(1, it.unsqueeze(1))  # gather the logprobs at sampled positions
        return it, sampleLogprobs

    def decode_sequence(self, seq):
        return utils.decode_sequence(self.vocab, seq)
