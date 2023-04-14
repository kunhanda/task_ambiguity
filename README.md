
# Overview

This repository contains code for the paper [Task Ambiguity in Humans and Language Models](https://arxiv.org/abs/2212.10711).

Within this repository is AmbiBench, a new benchmark of six ambiguously-specified classification tasks. The goal of AmbiBench is to construct a testbed of minimal complexity where we can control and measure the degree of ambiguity in various task specifications.

The code contains functionality to test language models on the three different AmbiBench settings discussed in the paper:
1.  task disambiguation using natural language instructions
2.  task disambiguation using multiple examples
3.  finetuning a model to generalize well in the face of ambiguity

# Setup

1.  create a virtualenv (https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)

2.  ``pip install -r requirements.txt``

3.  create a file named ``keys.py`` and create a variable named ``OPENAI_API_KEY = ‘your key goes here’``


# Running Experiments

Example command:

``main.py --shots=20 --model=’davinci’ --need_informative=False``

_It is currently only possible to use this codebase to run tests using the OpenAI API as tests done on other models in the paper used an internal API. If you desire to use AmbiBench with non-OpenAI models, please refer to the API documentation for that model and modify the neccessary information in ``keys.py`` and ``api_access.py``._

When calling _main.py_, you can add arguments specifying:
```python
type_1 (str) : {‘subject_location’, ‘religious_pronoun’, ‘propn_negation’}
type_2 (str) : {‘subject_location’, ‘religious_pronoun’, ‘propn_negation’}
type_3 (str): {‘subject_location’, ‘religious_pronoun’, ‘propn_negation’}
shots (int): n >= 0
model (str): if using OpenAI API, name of model to use (e.g. ‘text-davinci-002’)
format_1 (str): {‘arrow’, ‘qa’}
format_2 (str): {‘arrow’, ‘qa’}
need_instruction (bool): True if an instruction is required
need_informative (bool): True if the instruction should be an informative instruction (as opposed to an uninformative instruction)
verbose (bool): True if would like to see intermediate results when running tests
crfm (bool): True if the tests are run on the Stanford CRFM API (as opposed to OpenAI API)
prob_of_ambigous (float): The percentage of examples that should be ambiguous
togethercomputer (bool): True if generating a json to send to Stanford internal T0pp testing API
finetuning_control (bool): True if test is control test for finetuning (as opposed to ambiguous test)
```


To reproduce all tests discussed in the paper, only ``shots``, ``model``, ``need_informative``,  and ``finetuning_control`` need to be modified (for OpenAI models).


The ``all_tests = …`` line will also need to be modified.

## 1.  Task disambiguation using natural language instruction
Example command:
``main.py --shots=20 --need_informative=False --model=’davinci’``

For the arguments for the argparse defined in _main.py_, make sure that ``shots = 20``, ``need_informative = False``, and ``model`` is set to whatever model you want to run the test on.

Also, in ``main.py``, ensure that:

``all_tests = tester.run_two_feature_tests(args)``

## 2.  Task disambiguation using multiple examples
Example command:
``main.py --shots=1 --need_informative=False --model=’davinci’``

Make sure that ``shots = 1``, ``need_informative = True`` if running test with informative instructions and ``False`` if running test with uninformative instructions, and model is set to whatever model you want to test on.

Also, in ``main.py``, ensure that:

``all_tests = tester.run_two_feature_tests_with_two_set(args)``

## 3.  Finetuning a model to generalize well in the face of ambiguity
Example command:
``main.py --shots=20 --need_informative=False --model=’custom_finetuned_model’``

Make sure that ``shots = 20``, ``need_informative = False``.

If running the control experiments (finetuning on unambiguous data), set ``finetuning_control = True``. If running the ambiguous experiments, set ``finetuning_control = False``.

Then in ``tester.py``:

1.  in ``run_baseline_tests_for_finetuning``, make sure that ``salient_task_list`` contains only the tasks you want to finetune on. In our experiments, we withheld one construction_type pair (either ‘subject’ & ‘location’, ‘religious’ & ‘pronoun’, or ‘propn’ & ‘negation’) were withheld from salient_task_list.

2.  in ``run_finetuned_set``, ``salient_task_list`` contains only the two tasks withheld from ``salient_task_list`` in ``run_baseline_tests_for_finetuning``.

In ``main.py``,

first run ``tester.baseline_tests_for_finetuning(args)`` then ``all_tests = tester.run_finetuned_set(args)``

``run_baseline_tests_for_finetuning`` will only create the local file with which to finetune an OpenAI model. To finetune the model, follow the instructions on [https://beta.openai.com/docs/guides/fine-tuning](https://beta.openai.com/docs/guides/fine-tuning)

After finetuning and prior to running ``run_finetuned_set``, change ``model`` to the name of your finetuned model (provided by OpenAI API).

For all tests, set ``file_name`` to the path at which you want to save the results.

# Visualization
e.g:

``v = Visualizer(all_tests, args.needs_instruction)``
``v.visualize_accuracy()``

Create a new Visualizer object and call the function corresponding to the test you ran (docstrings for each function available in ``visualizer.py``). Generally, for (1), use ``visualize_accuracy``. And for (2), use ``visualize_accuracy_across_shots``. And for (3), use ``plot_individual_finetuning_performance_for_heldout``.


In ``visualizer.py`` you can set the output path for the generated figure in the last line of each function.

# Documenting import bits of Code

## `Prompt` class
```
Creates a prompt for the OpenAI API using by generating examples
    A Prompt consists of three Examples (the last one being called the query), metadata on each of those Examples, and in some cases, an instruction
    For example, a Prompt may look like:
        Instruction
        Example 1 {metadata}
        Example 2 {metadata}
        Query {metadata}
```

To generate examples, it determines the corresponding generator type given a `construction_type = {SubjectLocation; PropnNegation; ReligiousPronoun}`.

In the simple (i.e. non-salient) case,

# How to generate a set of examples and prompts?

To create a set examples, we wish to obtain a JSON file with queries and their expected completions. The examples should be constructed based on the configuration provided by arguments (e.g., `needs_informative = True`).

```
{
    "date": "YY-MM-DD_HH-mm",
    "configuration": {
        "arg1": "val1"
    },
    "examples": [
        {
            "query": "query text",
            "completion": "X"
        },
        {
            "query": "another query text",
            "completion": "Y"
        },
    ]

}

```


#
