# code from officital NN-Viterbi implementation by Alexander Richard https://github.com/alexanderrichard/NeuralNetwork-Viterbi/tree/master/utils (MIT License) - some modifications
# has MIT license
#!/usr/bin/python3

import numpy as np
import torch
import glob
import re


class LengthModel(object):
    def n_classes(self):
        return 0

    def score(self, length, label):
        return 0.0

    def max_length(self):
        return np.inf


class PoissonModel(LengthModel):
    def __init__(self, model, num_classes, sample_rate, max_length=2000, renormalize=True):
        super(PoissonModel, self).__init__()
        self.num_classes = num_classes
        if type(model) == str:
            if model.split(".")[-1] == "txt":
                self.mean_lengths = np.loadtxt(model)
            else:
                self.mean_lengths = torch.load(model)
        else:
            self.mean_lengths = np.ones((self.num_classes), dtype=np.float32)
        self.mean_lengths = (self.mean_lengths / sample_rate).round()
        self.max_len = max_length
        self.renormalize = renormalize
        self.poisson = np.zeros((max_length, self.num_classes))
        self.precompute_values()

    def precompute_values(self):
        # precompute normalizations for mean length model
        self.norms = np.zeros(self.mean_lengths.shape)
        if self.renormalize:
            self.norms = np.round(self.mean_lengths) * np.log(np.round(self.mean_lengths)) - np.round(self.mean_lengths)
            for c in range(len(self.mean_lengths)):
                logFak = 0
                for k in range(2, int(self.mean_lengths[c]) + 1):
                    logFak += np.log(k)
                self.norms[c] = self.norms[c] - logFak
        # precompute Poisson distribution
        self.poisson[0, :] = -np.inf  # length zero can not happen
        logFak = 0
        for l in range(1, self.max_len):
            logFak += np.log(l)
            self.poisson[l, :] = l * np.log(self.mean_lengths) - self.mean_lengths - logFak - self.norms

    def n_classes(self):
        return self.num_classes

    def score(self, length, label):
        if length >= self.max_len:
            return -np.inf
        else:
            return self.poisson[length, label]

    def max_lengths(self):
        return self.max_len

    def update_mean_lengths(self):
        self.mean_lengths = np.zeros((self.num_classes), dtype=np.float32)
        for label_count in self.buffer.label_counts:
            self.mean_lengths += label_count
        instances = np.zeros((self.num_classes), dtype=np.float32)
        for instance_count in self.buffer.instance_counts:
            instances += instance_count
        # compute mean lengths (backup to average length for unseen classes)
        self.mean_lengths = np.array([self.mean_lengths[i] / instances[i] if instances[i] > 0 else sum(self.mean_lengths) / sum(instances) for i in range(self.num_classes)])
        self.precompute_values()


class Grammar(object):

    # @context: tuple containing the previous label indices
    # @label: the current label index
    # @return: the log probability of label given context p(label|context)
    def score(self, context, label):  # score is a log probability
        return 0.0

    # @return: the number of classes
    def n_classes(self):
        return 0

    # @return sequence start symbol
    def start_symbol(self):
        return -1

    # @return sequence end symbol
    def end_symbol(self):
        return -2

    # @context: tuple containing the previous label indices
    # @return: list of all possible successor labels for the given context
    def possible_successors(context):
        return set()


# grammar that generates only a single transcript
# use during training to align frames to transcript
class SingleTranscriptGrammar(Grammar):

    def __init__(self, transcript, n_classes):
        self.num_classes = n_classes
        transcript = transcript + [self.end_symbol()]
        self.successors = dict()
        for i in range(len(transcript)):
            context = (self.start_symbol(),) + tuple(transcript[0:i])
            self.successors[context] = set([transcript[i]]).union( self.successors.get(context, set()) )

    def n_classes(self):
        return self.num_classes

    def possible_successors(self, context):
        return self.successors.get(context, set())

    def score(self, context, label):
        if label in self.possible_successors(context):
            return 0.0
        else:
            return -np.inf


# Viterbi decoding
class Viterbi(object):

    ### helper structure ###
    class TracebackNode(object):
        def __init__(self, label, predecessor, boundary = False):
            self.label = label
            self.predecessor = predecessor
            self.boundary = boundary

    ### helper structure ###
    class HypDict(dict):
        class Hypothesis(object):
            def __init__(self, score, traceback):
                self.score = score
                self.traceback = traceback
        def update(self, key, score, traceback):
            if (not key in self) or (self[key].score <= score):
                self[key] = self.Hypothesis(score, traceback)

    # @grammar: the grammar to use, must inherit from class Grammar
    # @length_model: the length model to use, must inherit from class LengthModel
    # @frame_sampling: generate hypotheses every frame_sampling frames
    # @max_hypotheses: maximal number of hypotheses. Smaller values result in stronger pruning
    def __init__(self, args, frame_sampling = 1, max_hypotheses = np.inf):
        mean_length_file = args.data_root_mean_duration + '/' + args.dataset + '/splits/train_split' + str(args.split) + '_mean_duration.txt'
        length_model = PoissonModel(mean_length_file, args.num_classes, args.sample_rate)
        self.grammar = None
        self.length_model = length_model
        self.frame_sampling = frame_sampling
        self.max_hypotheses = max_hypotheses

    # Viterbi decoding of a sequence
    # @log_frame_probs: logarithmized frame probabilities
    #                   (usually log(network_output) - log(prior) - max_val, where max_val ensures negativity of all log scores)
    # @return: the score of the best sequence,
    #          the corresponding framewise labels (len(labels) = len(sequence))
    #          and the inferred segments in the form (label, length)
    def decode(self, frame_probs, labels, n_classes):
        device = frame_probs.device
        transcript = labels[0, 1:-1] - 2
        transcript = [transcript[i].item() for i in range(transcript.shape[-1])]
        self.grammar = SingleTranscriptGrammar(transcript, n_classes)
        log_frame_probs = torch.log(frame_probs+1e-16).cpu().numpy()
        # if np.isinf(log_frame_probs).sum().item()!=0:
        #     print(1)
        # +(1e-16)
        log_frame_probs = log_frame_probs - np.max(log_frame_probs)
        assert log_frame_probs.shape[1] == self.grammar.n_classes()
        frame_scores = np.cumsum(log_frame_probs, axis=0) # cumulative frame scores allow for quick lookup if frame_sampling > 1
        # create initial hypotheses
        hyps = self.init_decoding(frame_scores)
        # decode each following time step
        for t in range(2 * self.frame_sampling - 1, frame_scores.shape[0], self.frame_sampling):
            hyps = self.decode_frame(t, hyps, frame_scores)
            self.prune(hyps)
        # transition to end symbol
        final_hyp = self.finalize_decoding(hyps)
        labels, segments = self.traceback(final_hyp, frame_scores.shape[0])
        labels = torch.tensor(np.array(labels)).unsqueeze(0).to(device)
        lengths = torch.tensor(np.array([s.length for s in segments])).to(device)
        if len(lengths) != len(transcript):
            lengths = []
            i = 0
            for t in transcript:
                if i < len(segments) and segments[i].label == t:
                    lengths.append(segments[i].length)
                    i = i + 1
                else:
                    lengths.append(0)
            lengths = torch.tensor(np.array(lengths)).to(device)
        duration_viterbi = lengths.unsqueeze(0) / lengths.sum()
        return duration_viterbi


    ### helper functions ###
    def frame_score(self, frame_scores, t, label):
        if t >= self.frame_sampling:
            # try:
            # return np.nan_to_num(frame_scores[t, label], neginf=0) - np.nan_to_num(frame_scores[t - self.frame_sampling, label], neginf=0)
            # # except RuntimeWarning:
            #     print( np.nan_to_num(frame_scores[t, label], neginf=0), frame_scores[t, label] , frame_scores[t - self.frame_sampling, label])
            return frame_scores[t, label] - frame_scores[t - self.frame_sampling, label]

        else:
            return frame_scores[t, label]

    def prune(self, hyps):
        if len(hyps) > self.max_hypotheses:
            tmp = sorted( [ (hyps[key].score, key) for key in hyps ] )
            del_keys = [ x[1] for x in tmp[0 : -self.max_hypotheses] ]
            for key in del_keys:
                del hyps[key]

    def init_decoding(self, frame_scores):
        hyps = self.HypDict()
        context = (self.grammar.start_symbol(),)
        for label in self.grammar.possible_successors(context):
            key = context + (label, self.frame_sampling)
            score = self.grammar.score(context, label) + self.frame_score(frame_scores, self.frame_sampling - 1, label)
            hyps.update(key, score, self.TracebackNode(label, None, boundary = True))
        return hyps

    def decode_frame(self, t, old_hyp, frame_scores):
        new_hyp = self.HypDict()
        for key, hyp in old_hyp.items():
            context, label, length = key[0:-2], key[-2], key[-1]
            # stay in the same label...
            if length + self.frame_sampling <= self.length_model.max_length():
                new_key = context + (label, length + self.frame_sampling)
                score = hyp.score + self.frame_score(frame_scores, t, label)
                new_hyp.update(new_key, score, self.TracebackNode(label, hyp.traceback, boundary = False))
            # ... or go to the next label
            context = context + (label,)
            for new_label in self.grammar.possible_successors(context):
                if new_label == self.grammar.end_symbol():
                    continue
                new_key = context + (new_label, self.frame_sampling)
                score = hyp.score + self.frame_score(frame_scores, t, label) + self.length_model.score(length, label) #+ self.grammar.score(context, new_label)
                new_hyp.update(new_key, score, self.TracebackNode(new_label, hyp.traceback, boundary = True))
        # return new hypotheses
        return new_hyp

    def finalize_decoding(self, old_hyp):
        final_hyp = self.HypDict.Hypothesis(-np.inf, None)
        for key, hyp in old_hyp.items():
            context, label, length = key[0:-2], key[-2], key[-1]
            context = context + (label,)
            score = hyp.score + self.length_model.score(length, label) #+ self.grammar.score(context, self.grammar.end_symbol())
            if score >= final_hyp.score:
                final_hyp.score, final_hyp.traceback = score, hyp.traceback
        # return final hypothesis
        return final_hyp

    def traceback(self, hyp, n_frames):
        class Segment(object):
            def __init__(self, label):
                self.label, self.length = label, 0
        traceback = hyp.traceback
        labels = []
        segments = [Segment(traceback.label)]
        while not traceback == None:
            segments[-1].length += self.frame_sampling
            labels += [traceback.label] * self.frame_sampling
            if traceback.boundary and not traceback.predecessor == None:
                segments.append( Segment(traceback.predecessor.label) )
            traceback = traceback.predecessor
        segments[0].length += n_frames - len(labels) # append length of missing frames
        labels += [hyp.traceback.label] * (n_frames - len(labels)) # append labels for missing frames
        return list(reversed(labels)), list(reversed(segments))


