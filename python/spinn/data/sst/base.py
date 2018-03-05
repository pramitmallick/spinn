# Not efficient, but good enough.


def lb_build(N):
    if N==2:
        return [0,0,1]
    else:
        return lb_build(N-1)+[0,1]

def rb_build(N):
    return [0]*N+[1]*(N-1)


def balanced_transitions(N):
    """
    Recursively creates a balanced binary tree with N
    leaves using shift reduce transitions.
    """
    if N == 3:
        return [0, 0, 1, 0, 1]
    elif N == 2:
        return [0, 0, 1]
    elif N == 1:
        return [0]
    else:
        right_N = N // 2
        left_N = N - right_N
        return balanced_transitions(left_N) + balanced_transitions(right_N) + [1]

def span_to_example(words,
        keep_fn=lambda x: True,
        convert_fn=lambda x: x,
        id='',
        transition_mode="default"):
    label = words[0][1]
    if not keep_fn(label):
        return None
    label = convert_fn(label)

    example = {}
    example["label"] = label
    example["sentence"] = " ".join(words)
    example["tokens"] = []
    example["transitions"] = []
    for index, word in enumerate(words):
        if word[0] != "(":
            if word == ")":
                # Ignore unary merges
                if words[index - 1] == ")":
                    example["transitions"].append(1)
            else:
                # Downcase all words to match GloVe.
                example["tokens"].append(word)
                example["transitions"].append(0)
    example["example_id"] = id
    if len(example["tokens"])>1:
        if transition_mode=="full_left":
            example["transitions"]=lb_build(len(example["tokens"]))
        elif transition_mode=="full_right":
            example["transitions"]=rb_build(len(example["tokens"]))
        elif transition_mode=="balanced":
            example["transitions"]=balanced_transitions(len(example["tokens"]))
    return example

def convert_unary_binary_bracketed_data(
        filename,
        keep_fn=lambda x: True,
        convert_fn=lambda x: x,
        top_node_only=False,
        transition_mode="default"):
    # Build a binary tree out of a binary parse in which every
    # leaf node is wrapped as a unary constituent, as here:
    #   (4 (2 (2 The ) (2 actors ) ) (3 (4 (2 are ) (3 fantastic ) ) (2 . ) ) )
    if top_node_only:
        print("SST eval mode: Preserving only top node label.")
    examples = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            stack = []
            words = line.replace(')', ' )')
            words = words.split(' ')
            if top_node_only:
                example = span_to_example(words, keep_fn, convert_fn, str(len(examples)))
                if example is not None:
                    examples.append(example)
            else:
                for index, word in enumerate(words):
                    if word[0] != "(":
                        if word == ")":
                            start = stack.pop()
                            example = span_to_example(words[start:index + 1], keep_fn, convert_fn, str(len(examples)), 
                                transition_mode=transition_mode)
                            if example is not None:
                                examples.append(example)
                    else:
                        stack.append(index)
    return examples
