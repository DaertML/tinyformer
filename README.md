## Tinyformer :microscope: :robot:
##### We are doing transformers without any Python dependency at Daert!

WIP repository with the objective to develop the code for inference for different LLM (Large Language Models) architectures based on the transformer. The main objective of the code developed in this repository, is not just to run inference in the transformer without dependencies, but to run it in a straight forward manner, without Object Oriented Programming, classes, functions, nested loops that run the code for each module... Just the plain operations.

The auxiliary functions are located in "functions.py", the weights for the models are located in "const.py", and the following are the different LLM Transformer architectures:
- vanilla_transformer.py: code from the original Transformer paper (https://arxiv.org/pdf/1706.03762.pdf).

Run the following to run a toy example:
~~~bash
$ python3 vanilla_transformer.py
~~~