# PsychFormers: Tools for calculating psycholinguistically-relevant metrics of language statistics using transformer language models

This repository contains a command-line Python script (`psychformers.py`) that allows the user to use transformer neural network language models (via the `transformers` Python package) to calculate metrics that are relevant to psycholinguistic experiments. Currently, only the `surprisal` metric is supported.

An additional script (`psychformers_gpt3.py`) is also included that allows the user to calculate the `surprisal` metric for a given word or sequence of words using GPT-3.

## Metrics
Psychformers currently supports the following metrics:

* `surprisal`

### Surprisal
Surprisal is the negative log-probability of a token or sequence of tokens, as calculated by a language model. That is, for a token $t$, the surprisal is given by:

$$-\log(p_t)$$

The surprisal of a word, as calculated by a language model, has been found to correlate with behavioral metrics of processing difficulty such as reading time (Smith and Levy, 2013) as well as neural metrics such as the N400 (Frank et al., 2015).

For sequences of tokens (either sequences of words or individual words that are made up of more than a single token), Psychformers calculates the surprisal by predicting the first token based on the context only, then adding the true first token to the context and predicting the next token, and so on for all tokens. The surprisal of the sequence is operationalized as the sum of the surprisals of all the tokens.


## How to use

### How to run

PsychFormers is a Python script that can be run from the command line. For example, assume we want to calculate the `surprisal` for the stimuli in a file `stimuli.stims` using `gpt2`, and output it to `output_folder`, we could run the script in the following way:

```
python psychformers.py -i stimuli.txt -o output_folder -m gpt2 -t surprisal
```

PsychFormers would run the stimuli and output the results to  `output_folder/stimuli.surprisal.gpt2.causal.output`.

Required and optional arguments for `psychformers.py` will be explained below.

### Input Format
The format of the input files must follow a strict form. Firstly, the input file must be a plain text file. Each individual stimulus item should be provided on a separate line, with the target&mdash;that is, the word or sequence of words for which the metric is to be calculated&mdash;surrounded by asterisks (`*`). For example, such a file may contain lines like the following:

```
...
The metric to be calculated is *surprisal*.
Other *metrics* could also be calculated.
...
```
In this case, PsychFormers will calculate the relevant metric(s) for the words `surprisal` and `metrics` based on the context of the line on which each.

For clarity, stimulus files have been given the `.stims` extension, but as long as the file is a plain text file, any extension should work with PsychFormers.

### Output Format
PsychFormers creates one or more output files, where each filename has the format: `[input_filename].[metric].[model].[model_type].output`. 

* `input_filename` is the full name of the input file before its extension.
* `metric` is the metric calculated.
* `model` is the model used.
* `model_type` is one of three `causal` (autoregressive), `masked`, or `causal_mask` (a masked language model with a causal mask).


The file itself is a tab-separated values file, with five columns: 

* `FullSentence`: The full stimulus as written in the input file (with asterisks removed).
* `Sentence`: The stimulus, encoded and decoded; useful for testing how a stimulus has been processed by a given model.
* `TargetWords`: The target, i.e., the words or words for which the metric has been calculated.
* The name of the metric (e.g. `Surprisal`).
* `NumTokens`: The number of tokens making up the target.



### Required arguments
* `--stimuli` (`-i`) or `--stimuli_list` (`-ii`): The input(s) to the language models. Note that a `--stimuli_list` argument will override a `--stimuli` argument.
    * `--stimuli` (`-i`): The path to a file containing the stimuli to be run through the language models (in the format described above).
    * `--stimuli_list` (`-ii`): The path to a file listing the paths to one or more input files, separated by line.
* `--output_directory` (`-o`): The output directory.
* `--model` (`-m`) or `--model_list` (`-mm`): The model(s) to use to calculate the metrics.  Note that a `--model_list` argument will override a `--model` argument.
    * `--model` (`-m`): The model to be used to calculate the metric(s) specified. This must be the name of the model as it appears in the [Hugging Face Model Hub](https://huggingface.co/models).
    * `--model_list` (`-mm`): The path to a file listing the models to be used (in the format specified above), separated by line.
* `--task` (`-t`) or `--task_list` (`-tt`): The metric(s) to be calculated by each model for each stimulus item. Note that a `--task_list` argument will override a `--task` argument.
    * `--task` (`-t`): The metric to be calculated. Currently only supports `surprisal`.
    * `--task_list` (`-tt`): The path to a file listing the metrics to be calculated, separated by line.

### Optional Arguments
* `--following_context` (`-f`): Whether or not to include the following context&mdash;that is, anything following the target&mdash;when calculating surprisal. The default is `False`; including this argument will switch this to `True`. Note that if you include this argument, only the results for masked language models will be affected, as it is not possible for autoregressive (`causal`) models to use the following context for prediction.
* `--primary_decoder` (`-d`): For models with both `masked` and `causal` versions, specify which one to use as the default&mdash;either `masked` or `causal` (default is `masked`). This is mostly used to select causally-masked versions of masked language models.
* `--use_cpu` (`-cpu`): Run models on CPU even if CUDA is supported and available.


## PsychFormers for GPT-3

PsychFormers for GPT-3 (`psychformers_gpt3.py`) can be used to calculate surprisal in the same way using GPT-3. The arguments differ in the following way:
* `--model` (`-m`) OR `-model_list` (`-mm`): uses the name of the specific GPT-3 model as listed in OpenAI API documentation rather than the name of the model on the Hugging Face Model Hub.
* `--key` (`-k`): a new argument that allows the user to input their API key from the command-line. An alternative is to add this to the script on line 9.
* Because GPT-3 is an autoregressive (aka `causal`) language model, the `--primary_decoder`(`-d`) and `--following_context` (`-f`) arguments have been removed as possible arguments.
* Because only `surprisal` is calculated, the `--task` (`-t`) and `--task_list` (`-tt`) arguments have been removed as possible arguments.


Additionally, GPT-3 does not appear to use a beginning-of-sequence (or equivalent) token; thus, it is important to note that it is not possible to calculate the surprisal of the first word in a sequence.

## Requirements
PsychFormers was written for `Python 3.8` and requires the `pytorch` and `transformers` packages. PsychFormers for GPT-3 also requires the `openai` package and an OpenAI API Key.

## How to cite

The first version of this code was released [here](https://github.com/jmichaelov/italian-zero-anaphora-prediction) to accompany [this paper](https://arxiv.org/abs/2208.14554) (accepted at COLING 2022). For now, if you use this code, please cite the paper:

```
@article{michaelov_2022_AnaphoricZeroPronouns,
  title={Do language models make human-like predictions about the coreferents of Italian anaphoric zero pronouns?},
  author={Michaelov, James A. and Bergen, Benjamin K.},
  journal={arXiv preprint arXiv:2208.14554},
  year={2022}
}
```

## References
* Smith, N. J., & Levy, R. (2013). [The effect of word predictability on reading time is logarithmic](https://doi.org/10.1016/j.cognition.2013.02.013). Cognition, 128(3), 302-319.
* Frank, S. L., Otten, L. J., Galli, G., & Vigliocco, G. (2015). [The ERP response to the amount of information conveyed by words in sentences](https://doi.org/10.1016/j.bandl.2014.10.006). Brain and language, 140, 1-11.
