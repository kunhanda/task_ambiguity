import numpy as np
import pandas as pd


class MetricWrangler:
    """
    Isolates the logits,converts logits to probabilities, gets the average probability, and performs transformations on
    the dataframe the output from the API to make the results more readable and visualizable

    For example, given an output from the API, this class (in order):
    For each Prompt sent to the API
        1) isolates the logits and converts them to probs
        2) turns those probs into percentages
        3) updates the list of probs for all Prompts
    For all Prompts for each unique construction_type:
        1) calculates the average probs
        2) constructs a pd.Dat  aFrame containing all relevant info from all the outputs

    Attributes:
        test_example_prob (list(float)): a list of all the probabilities for the final label (query)
        final_probs_list (list(float)): a list of the all the probailities for all the alternatives for the query
        accuracies (list(float)): a list of the accuracies for the query (either 0 or 1 for each query)
    """

    def __init__(self):
        self.test_example_prob = []
        self.final_probs_list = []
        self.accuracies = []

    def label_probs(self, output_df, generate_instruction):
        """
        Isolate the probs for the X/Y labels from the output in a dataframe

        Args:
            output_df (pd.DataFrame): the data outputted by the OPENAI API
            generate_instruction (bool): a boolean to determine if an instruction should be generated or not

        Returns:
            label_df (pd.DataFrame): a DataFrame containing only neeccessary information from the output: label tokens (X/Y)
                and their corresponding probabilities
        """

        # removes all rows not containig the label tokens ('X' or 'Y')
        label_df = output_df.loc[
            (output_df["tokens"].str.strip(" ").str.strip("'") == "X")
            | (output_df["tokens"].str.strip(" ").str.strip("'") == "Y")
        ]

        label_df["top_k_probs"] = label_df["top_logprobs"].apply(
            lambda row: self.as_percentages(row)
        )
        label_df.drop(columns=["top_logprobs"], inplace=True)

        # removes label token probabilities for tokens in the instruction (as the instruction mentioned 'X' and 'Y')
        if generate_instruction:
            label_df = label_df.iloc[2:, :]

        # removes irrelevant column
        label_df = label_df.reset_index()
        label_df = label_df.drop(["index"], axis=1)

        label_df["example_number"] = label_df.index + 1

        self.update_accuracy(label_df)

        label_df["%"] = label_df.apply(
            lambda row: self.recalc_percentage(
                row["tokens"].strip(), row["top_k_probs"], row["%"]
            ),
            axis=1,
        )

        return label_df

    def recalc_percentage(self, token, token_dict, curr_percentage):
        if token in token_dict:
            return token_dict[token]
        else:
            return curr_percentage

    def construct_test_example_df(self, test_examples, test_examples_output_df):
        """
        Constructs the complete DataFrame for the queries including both the input and output information

        Args:
            test_examples (list(Example)): a list of all the queries and their relevant information
            test_examples_output_df (pd.DataFrame): a DataFrame of all the outputs obtained from the API for the queries
        Returns:
            test_examples_complete_df (pd.DataFrame): a DataFrame with only the relevant information on label tokens ('X'/'Y'),
            meta data on their corresponding examples and their corresponding probabilities
        """
        test_examples_input_df = pd.DataFrame.from_records(
            e.as_dict() for e in test_examples
        )
        test_examples_complete_df = pd.concat(
            [test_examples_input_df, test_examples_output_df], axis=1, join="inner"
        )

        return test_examples_complete_df

    def as_percentages(self, final_logits):
        return {keys: 100 * np.exp(vals) for keys, vals in final_logits.items()}

    def append_to_list(self, final_percentages):
        self.final_probs_list.append(final_percentages)

    def update_accuracy(self, label_df):
        """
        Checks whether the most recent query was classified correctly and appends the result to self.accuracies

        Args:
            probs_df (pd.DataFrame): a dataframe of the probabilities for the query
            label_percentages (dict): a dictionary of the top 4 probabilities from the API for the query
        Returns:
            ret (int): 1 if the model correctly predicted the output and 0 otherwise
        """
        label_df["accurate"] = label_df.apply(
            lambda row: self.check_accuracy(row["top_k_probs"], row["tokens"].strip()),
            axis=1,
        )

    def check_accuracy(self, label_percentages, correct_label):
        label_percentages = self.combine_keys(label_percentages)
        ret = 1
        if max(label_percentages, key=label_percentages.get) != correct_label:
            ret = 0
        self.accuracies.append(ret)
        return ret

    def combine_keys(self, og_d):
        """
        Combines probabilities for keys which are correct but contain extraneous spaces, so wouldn't be 'graded' as correct

        Args:
            og_d: original 'graded' dataframe
        Returns:
            og_d: updated dataframe with combined probabilities for X/Y keys
        """
        new_d = og_d.copy()
        if "X" in new_d:
            del new_d["X"]
        if "Y" in new_d:
            del new_d["Y"]
        new_d = {k.translate({32: None}): v for k, v in new_d.items()}

        if "X" in new_d:
            og_d["X"] = og_d.get("X", 0) + new_d.get("X", 0)
        if "Y" in new_d:
            og_d["Y"] = og_d.get("Y", 0) + new_d.get("Y", 0)

        if " X" in og_d:
            del og_d[" X"]
        if " Y" in og_d:
            del og_d[" Y"]

        return og_d
