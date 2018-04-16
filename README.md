# ListOps
This is the source code for the paper, "ListOps: A Diagnostic Dataset for Latent Tree Learning" by Nangia and Bowman, 2018.

It is built on a fairly large and unwieldy [codebase](https://github.com/stanfordnlp/spinn) that was prepared for the paper [A Fast Unified Model for Sentence Parsing and Understanding](https://arxiv.org/abs/1603.06021). The master branch is still under active development for other projects and may be buggy.

### ListOps Data
The ListOps dataset is released as text files and can be downloaded [here](https://github.com/nyu-mll/spinn/tree/listops-release/python/spinn/data/listops). 


### Data Generation
The data generation script is [make_data.py](https://github.com/nyu-mll/spinn/blob/listops-release/python/spinn/data/listops/make_data.py). Variables such as maximum tree-depth (MAX_DEPTH) can be changed to generate variations on ListOps, additional mathematical operations can also be added.

### Installation
If you want to run experiments with the models used in the paper, follow these steps for installation.

Requirements:

- Python 3.6
- PyTorch 0.3
- Additional dependencies listed in python/requirements.txt

Install PyTorch based on instructions online: http://pytorch.org

Install the other Python dependencies using the command below.

    python3 -m pip install -r python/requirements.txt

### Running the code

The main executables for the ListOps experiments in the paper are [supervised_classifier.py](https://github.com/mrdrozdov/spinn/blob/master/python/spinn/models/supervised_classifier.py) and [rl_classifier.py](https://github.com/mrdrozdov/spinn/blob/master/python/spinn/models/rl_classifier.py). The flags specfiy the model type and the hyperparameters. You can specify gpu usage by setting `--gpu` flag greater than or equal to 0. Uses the CPU by default.

Here is a sample command that runs a CPU training run for an RNN, training and testing on ListOps.

    PYTHONPATH=spinn/python \
        python3 -m spinn.models.supervised_classifier --data_type listops \
        --model_type RNN --training_data_path spinn/data/listops/train_d20s.tsv \
        --eval_data_path spinn/data/listops/test_d20s.tsv \ 
        --word_embedding_dim 128 --model_dim 128 --seq_length 100 \ 
        --eval_seq_length 3000 --mlp_dim 16 --num_mlp_layers 2 \
        --optimizer_type Adam --experiment_name RNN_samplerun

## Log Analysis

This project contains a handful of tools for easier analysis of your model's performance.

For one, after a periodic number of batches, some useful statistics are printed to a file specified by `--log_path`. This is convenient for visual inspection, and the script [parse_logs.py](https://github.com/nyu-mll/spinn/blob/master/scripts/parse_logs.py) is an example of how to easily parse this log file. 

The script [parse_comparison.py](https://github.com/nyu-mll/spinn/blob/master/scripts/parse_comparison.py) generates statistics about the quality of generated parses. It takes `.report` files as input. These files are generated during model evaluation if the `--write_eval_reports` flag is set to `True`. Here is a sample command,

    python scripts/parse_comparison.py --data_type listops --main_data_path python/spinn/data/listops/test_d20s.tsv --main_report_path_template RNN_samplerun.eval_set_0.report

## Contributing

If you're interested in proposing a change or fix to SPINN, please submit a Pull Request. In addition, ensure that existing tests pass, and add new tests as you see appropriate. To run tests, simply run this command from the root directory:

    nosetests python/spinn/tests

## License

Copyright 2018, New York University

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


