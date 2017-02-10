import itertools

import numpy as np
from spinn import util

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from spinn.util.blocks import BaseSentencePairTrainer, Reduce
from spinn.util.blocks import LSTMState, Embed, MLP
from spinn.util.blocks import bundle, unbundle, to_cpu, to_gpu, treelstm, lstm
from spinn.util.blocks import get_h, get_c
from spinn.util.misc import Args, Vocab, Example

from spinn.fat_stack import BaseModel, SentenceModel, SentencePairModel
from spinn.fat_stack import SPINN


T_SKIP   = 2
T_SHIFT  = 0
T_REDUCE = 1


class SentencePairTrainer(BaseSentencePairTrainer): pass


class SentenceTrainer(SentencePairTrainer): pass


class RLSPINN(SPINN):
    def predict_actions(self, transition_output, cant_skip):
        if self.training:
            transition_dist = F.softmax(transition_output)
            transition_dist = transition_dist.data.cpu().numpy()
            sampled_transitions = np.array([T_SKIP for _ in self.bufs], dtype=np.int32)
            sampled_transitions[cant_skip] = [np.random.choice(self.choices, 1, p=t_dist)[0] for t_dist in transition_dist[cant_skip]]
            transition_preds = sampled_transitions
        else:
            transition_dist = F.log_softmax(transition_output)
            transition_dist = transition_dist.data.cpu().numpy()
            transition_preds = transition_dist.argmax(axis=1)
        return transition_preds


class RLBaseModel(BaseModel):

    optimize_transition_loss = False

    def build_spinn(self, args, vocab, use_skips):
        return RLSPINN(args, vocab, use_skips=use_skips)

    def build_reward(self, logits, target):
        # Zero One Loss.
        rewards = torch.eq(logits.max(1)[1], target).float()

        # TODO: Cross Entropy Reward

        return rewards

    def reinforce(self, rewards):
        t_preds, t_logits, t_given, t_mask = self.spinn.get_statistics()

        if self.use_sentence_pair:
            # Handles the case of SNLI where each reward is used for two sentences.
            rewards = torch.cat([rewards, rewards], 0)

        # Expand rewards.
        if not self.spinn.use_skips:
            rewards = rewards.index_select(0, torch.from_numpy(t_mask).long())
        else:
            raise NotImplementedError

        log_p_action = torch.cat([t_logits[i, p] for i, p in enumerate(t_preds)], 0)

        rl_loss = -1. * torch.sum(log_p_action * Variable(rewards, volatile=log_p_action.volatile))
        rl_loss /= log_p_action.size(0)

        return rl_loss

    def output_hook(self, output, sentences, transitions, y_batch=None):
        if not self.training:
            return

        logits = F.softmax(output).data
        target = torch.from_numpy(y_batch).long()

        # Get Reward.
        rewards = self.build_reward(logits, target)

        # Assign REINFORCE output.
        self.rl_loss = self.reinforce(rewards)


class SentencePairModel(RLBaseModel):

    def build_example(self, sentences, transitions):
        batch_size = sentences.shape[0]

        # Build Tokens
        x_prem = sentences[:,:,0]
        x_hyp = sentences[:,:,1]
        x = np.concatenate([x_prem, x_hyp], axis=0)

        # Build Transitions
        t_prem = transitions[:,:,0]
        t_hyp = transitions[:,:,1]
        t = np.concatenate([t_prem, t_hyp], axis=0)

        example = Example()
        example.tokens = to_gpu(Variable(torch.from_numpy(x), volatile=not self.training))
        example.transitions = t

        return example

    def run_spinn(self, example, use_internal_parser=False, validate_transitions=True):
        state_both, transition_acc, transition_loss = super(SentencePairModel, self).run_spinn(
            example, use_internal_parser, validate_transitions)
        batch_size = len(state_both) / 2
        h_premise = get_h(torch.cat(state_both[:batch_size], 0), self.hidden_dim)
        h_hypothesis = get_h(torch.cat(state_both[batch_size:], 0), self.hidden_dim)
        return [h_premise, h_hypothesis], transition_acc, transition_loss


class SentenceModel(RLBaseModel):

    def build_example(self, sentences, transitions):
        batch_size = sentences.shape[0]

        # Build Tokens
        x = sentences

        # Build Transitions
        t = transitions

        example = Example()
        example.tokens = to_gpu(Variable(torch.from_numpy(x), volatile=not self.training))
        example.transitions = t

        return example

    def run_spinn(self, example, use_internal_parser=False, validate_transitions=True):
        state, transition_acc, transition_loss = super(SentenceModel, self).run_spinn(
            example, use_internal_parser, validate_transitions)
        h = get_h(torch.cat(state, 0), self.hidden_dim)
        return [h], transition_acc, transition_loss