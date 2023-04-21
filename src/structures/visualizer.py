import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class Visualizer:
    """
    Makes either barplot or lineplots using seaborn to visualize the results of the tests.

    Attributes:
        all_test_df (pd.DataFrame): dataframe containing outputs from all tests to be visualized
        needs_instruction (bool): True if tests required instructions, False otherwise

    """

    def __init__(self, all_test_df, needs_instruction):
        self.all_test_df = all_test_df
        self.needs_instruction = needs_instruction

    def visualize_probs(self):
        """
        Make a bar plot of the P(correct answer) across different construction and format types
        """
        sns.set_theme(style="whitegrid")

        plot = sns.catplot(
            data=self.all_test_df,
            kind="bar",
            x="salient_task",
            y="%",
            hue="format_type",
            ci=95,
            palette="dark",
            alpha=0.6,
        )
        plot.despine(left=True)
        plt.ylim(0, 100)
        plot.set_axis_labels("Salient Task", "P(correct answer)")
        plot.legend.set_title("Format Type")

        plt.savefig("")

    def visualize_accuracy(self):
        """
        Make a bar plot of the accuracy across different construction and format types
        """
        sns.set_theme(style="whitegrid")

        plot = sns.catplot(
            data=self.all_test_df,
            kind="bar",
            x="salient_task",
            y="accurate",
            hue="format_type",
            ci=95,
            palette="dark",
            alpha=0.6,
        )
        plot.despine(left=True)
        plt.ylim(0, 1.0)
        plot.set_axis_labels("Salient Task", "Accuracy")
        plot.legend.set_title("Format Type")

        plt.savefig("")

    def visualize_probs_across_shots(self, tests_df):
        """
        Make a line plot of the probability across different construction and format types
        """
        sns.set_theme(style="whitegrid")
        tests_df = tests_df[["salient_task", "format_type", "example_number", "%"]]
        sns.relplot(
            kind="line",
            data=tests_df,
            x="example_number",
            y="%",
            hue="format_type",
            col="salient_task",
            col_wrap=3,
        )

        plt.savefig("")

    def visualize_accuracy_across_shots(self, tests_df):
        """
        Make a line plot of the accuracy across different construction and format types
        """
        sns.set_theme(style="whitegrid")

        tests_df = tests_df[
            ["salient_task", "format_type", "example_number", "accurate"]
        ]
        sns.relplot(
            kind="line",
            data=tests_df,
            x="example_number",
            y="accurate",
            hue="format_type",
            col="salient_task",
            col_wrap=3,
        )

        plt.savefig("")

    def plot_individual_finetuning_performance_for_heldout(
        heldout_task_1, heldout_task_2, d_reg, d_i, control, ambig
    ):
        """
        Creates a lineplot for an individual heldout salient task pair

        Parameters:
            heldout_task_1 (str): first task heldout when finetuing
            heldout_task_2 (str): second task heldout when finetuning
            d_reg (pd.DataFrame):  DataFrame for davinci 20-examples test (task disambiguation using multiple examples)
            d_i (pd.DataFrame): DataFrame for text-davinci-002 20-examples test (task disambiguation using multiple examples)
            control (pd.DataFrame): DataFrame from control finetuning test
            ambig (pd.DataFrame): DataFrame from ambiguous finetuning test
        Returns:
            None
        """

        a = heldout_task_1
        b = heldout_task_2

        davinci_regular = d_reg
        davinci_instruct = d_i
        two_feature_alternating = ambig
        one_feature_alternating = control
        two_feature_alternating["Model"] = "davinci finetuned (ambiguous)"
        one_feature_alternating["Model"] = "davinci finetuned (control)"
        davinci_regular["Model"] = "davinci"
        davinci_instruct["Model"] = "text-davinci-002"

        selected_models = pd.concat([davinci_instruct, davinci_regular])
        selected_models = selected_models[
            (selected_models["salient_task"] == a)
            | (selected_models["salient_task"] == b)
        ]

        sns.set_theme(style="darkgrid")
        selected_models = pd.concat(
            [selected_models, one_feature_alternating, two_feature_alternating]
        )
        selected_models = selected_models.reset_index(drop=True)
        selected_models = selected_models[
            ["salient_task", "format_type", "example_number", "accurate", "Model"]
        ]
        plot = sns.lineplot(
            data=selected_models,
            x="example_number",
            y="accurate",
            hue="Model",
            palette="rocket",
        )
        plt.ylim(0, 1.0)
        plot.set(xlabel="Example Number", ylabel="Accuracy")
        plt.legend(title="Model")
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

        plt.savefig("")
