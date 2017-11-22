#!/usr/bin/env python

import math
import json
import codecs

SENTENCE_PAIR_DATA = True
FIXED_VOCABULARY = None
PADDING_TOKEN = "_PAD"

LABEL_MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
    # Used in the unlabeled test set---needs to map to some arbitrary label.
    "hidden": 0,
}


def full_tokens(tokens):
    """
    Pads a sequence of tokens from the left so the resulting
    length is a factor of two.
    """
    target_length = int(2 ** math.ceil(math.log(len(tokens), 2)))
    padding_length = target_length - len(tokens)
    tokens = [PADDING_TOKEN] * padding_length + tokens
    return tokens


def full_transitions(N):
    """
    Recursively creates a full binary tree of with N
    leaves using shift reduce transitions.
    """

    N = float(N)

    # Can not have a single word sentence.
    assert N > 1

    # Constrain to full binary trees.
    assert math.log(N, 2) % 1 == 0, \
        "Bad value. N={}".format(N)

    if math.log(N, 2) == 1:
        return [0, 0, 1]

    return full_transitions(N/2) + full_transitions(N/2) + [1]


def convert_binary_bracketing_full(parse, lowercase=False):
    tokens, transitions = convert_binary_bracketing(parse, lowercase)
    if len(tokens) > 1:
        tokens = full_tokens(tokens)
        transitions = full_transitions(len(tokens))
    return tokens, transitions


def convert_binary_bracketing(parse, lowercase=False):
    transitions = []
    tokens = []

    for word in parse.split(' '):
        if word[0] != "(":
            if word == ")":
                transitions.append(1)
            else:
                # Downcase all words to match GloVe.
                if lowercase:
                    tokens.append(word.lower())
                else:
                    tokens.append(word)
                transitions.append(0)

    return tokens, transitions


def load_data(path, lowercase=False, choose=lambda x: True, full=False):
    convert = convert_binary_bracketing_full if full else convert_binary_bracketing
    print "Loading", path
    examples = []
    failed_parse = 0
    with codecs.open(path, encoding='utf-8') as f:
        for line in f:
            try:
                line = line.encode('UTF-8')
            except UnicodeError as e:
                print "ENCODING ERROR:", line, e
                line = "{}"
            loaded_example = json.loads(line)
            if loaded_example["gold_label"] not in LABEL_MAP:
                continue

            if not choose(loaded_example):
                continue

            example = {}
            example["label"] = loaded_example["gold_label"]
            example["premise"] = loaded_example["sentence1"]
            example["hypothesis"] = loaded_example["sentence2"]
            example["example_id"] = loaded_example.get('pairID', 'NoID')
            if loaded_example["sentence1_binary_parse"] and loaded_example["sentence2_binary_parse"]:
                (example["premise_tokens"], example["premise_transitions"]) = convert(
                    loaded_example["sentence1_binary_parse"], lowercase=lowercase)
                (example["hypothesis_tokens"], example["hypothesis_transitions"]) = convert(
                    loaded_example["sentence2_binary_parse"], lowercase=lowercase)
                examples.append(example)
            else:
                failed_parse += 1
    if failed_parse > 0:
        print(
            "Warning: Failed to convert binary parse for {} examples.".format(failed_parse))
    return examples


if __name__ == "__main__":
    # Demo:
    examples = load_data('snli-data/snli_1.0_dev.jsonl')
    print examples[0]
