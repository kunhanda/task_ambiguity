import time

import pandas as pd

from src.query_pipeline import QueryPipeline


class Tester:
    def run_test(
        self,
        construction_type,
        shots,
        model,
        construction_format,
        crfm,
        queries,
        needs_instruction,
        verbose,
        needs_informative,
        include_ambiguous_examples,
        prob_of_ambiguous,
        togethercomputer,
        for_finetuning,
        finetuning_control,
        salient_task=None,
    ):
        """
        Runs a single test which consists of a single query to the API with one Prompt
        Args:
            construction_type (str): type of construction to generate (options included in main)
            shots (int): number of examples to include in Prompt (including query)
            model (str): model on which to run test (options included in main)
            construction_format (str): format of constructions (options included in main)
            crfm (bool): if True runs on Stanford's CRFM API, if False runs with OpenAI API
            queries (int): number of queries/tests for the current construction_type + construction_format combination
            needs_instruction (bool): if True adds instructions to the Prompt
            verbose (bool): to print out detailed information on all Prompts while running
            needs_informative (bool): if True if needs informative instruction, False otherwise
            include_ambiguous_examples (bool): if True includes ambiguous examples in the Prompt, False otherwise
            prob_of_ambiguous (float): probability of including an ambiguous example in the Prompt
            togethercomputer (bool): if True runs the test with the TogetherComputer API, False otherwise
            for_finetuning (bool): if True runs the test for finetuning, False otherwise
            finetuning_control (bool): True if running finetuning control tests
            salient_task (str): if not None, the salient task for the current test

        Returns:
            test_df (pd.DataFrame): DataFrame containing all relevant information obtained from running the test
        """
        test = QueryPipeline(construction_type, shots, model, construction_format, crfm)
        test_df = test.run_pipeline(
            queries=queries,
            needs_instruction=needs_instruction,
            verbose=verbose,
            needs_informative=needs_informative,
            include_ambiguous_examples=include_ambiguous_examples,
            salient_task=salient_task,
            prob_of_ambiguous=prob_of_ambiguous,
            togethercomputer=togethercomputer,
            finetuning_control=finetuning_control,
            for_finetuning=for_finetuning,
        )
        return test_df

    def run_two_feature_tests(self, args):
        """
        Runs all standard tests which are two-feature tests {'subject_location', 'religious_pronoun', 'propn_negation'}

        Args:
            args (ArgumentParser.args): command line arguments from main
        Returns:
            all_tests (pd.DataFrame): DataFrame containg the relevant information from all Prompts queried
        """
        all_tests = pd.DataFrame()

        construction_formats_list = [args.format_2, args.format_1]
        salient_tasks_list = [
            "subject",
            "location",
            "religious",
            "negation",
            "propn",
            "pronoun",
        ]

        construction_types_map = {
            "location": "subject_location",
            "subject": "subject_location",
            "religious": "religious_pronoun",
            "pronoun": "religious_pronoun",
            "propn": "propn_negation",
            "negation": "propn_negation",
        }

        all_tests = pd.DataFrame()
        for cf in construction_formats_list:
            for st in salient_tasks_list:
                for _ in range(
                    3
                ):  # need loop in order to not overload API and stay within OpenAI constraints
                    curr_test = self.run_test(
                        construction_type=construction_types_map[st],
                        shots=args.shots,
                        model=args.model,
                        construction_format=cf,
                        crfm=args.crfm,
                        queries=20,
                        needs_instruction=args.needs_instruction,
                        verbose=args.verbose,
                        needs_informative=args.needs_informative,
                        include_ambiguous_examples=args.include_ambiguous_examples,
                        salient_task=st,
                        prob_of_ambiguous=args.prob_of_ambiguous,
                        togethercomputer=args.togethercomputer,
                        for_finetuning=False,
                        finetuning_control=False,
                    )

                    all_tests = pd.concat([all_tests, curr_test], ignore_index=True)
                    if not args.crfm and not args.togethercomputer:
                        time.sleep(60)

        return all_tests

    def run_two_feature_tests_with_two_set(self, args):
        """
        Runs all standard tests which are two-feature tests {'subject_location', 'religious_pronoun', 'propn_negation'}

        Args:
            args (ArgumentParser.args): command line arguments from main
        Returns:
            all_tests (pd.DataFrame): DataFrame containg the relevant information from all Prompts queried
        """
        all_tests = pd.DataFrame()
        construction_types_list = [args.type_1, args.type_2, args.type_3]
        construction_formats_list = [args.format_2, args.format_1]

        all_tests = pd.DataFrame()
        for cf in construction_formats_list:
            for ct in construction_types_list:
                for _ in range(
                    3
                ):  # need loop in order to not overload API and stay within OpenAI constraints
                    curr_test = self.run_test(
                        construction_type=ct,
                        shots=args.shots,
                        model=args.model,
                        construction_format=cf,
                        crfm=args.crfm,
                        queries=20,
                        needs_instruction=args.needs_instruction,
                        verbose=args.verbose,
                        needs_informative=args.needs_informative,
                        include_ambiguous_examples=args.include_ambiguous_examples,
                        salient_task=None,
                        prob_of_ambiguous=args.prob_of_ambiguous,
                        togethercomputer=args.togethercomputer,
                        for_finetuning=True,
                        finetuning_control=False,
                    )

                    all_tests = pd.concat([all_tests, curr_test], ignore_index=True)
                    if not args.crfm:
                        time.sleep(60)

        return all_tests

    def run_baseline_tests_for_finetuning(self, args):
        """
        Runs all standard tests which are two-feature tests {'subject_location', 'religious_pronoun', 'propn_negation'}

        Args    :
            args (ArgumentParser.args): command line arguments from main
        Returns:
            all_tests (pd.DataFrame): DataFrame containg the relevant information from all Prompts queried
        """
        all_tests = pd.DataFrame()

        construction_formats_list = [args.format_2, args.format_1]
        salient_tasks_list = ["religious", "pronoun", "propn", "negation"]

        construction_types_map = {
            "location": "subject_location",
            "subject": "subject_location",
            "religious": "religious_pronoun",
            "pronoun": "religious_pronoun",
            "propn": "propn_negation",
            "negation": "propn_negation",
        }

        all_tests = pd.DataFrame()
        for cf in construction_formats_list:
            for st in salient_tasks_list:
                for _ in range(2):
                    for i in range(
                        3, 20
                    ):  # need loop in order to not overload API and stay within OpenAI constraints
                        curr_test = self.run_test(
                            construction_type=construction_types_map[st],
                            shots=i,
                            model=args.model,
                            construction_format=cf,
                            crfm=args.crfm,
                            queries=1,
                            needs_instruction=args.needs_instruction,
                            verbose=args.verbose,
                            needs_informative=args.needs_informative,
                            include_ambiguous_examples=args.include_ambiguous_examples,
                            salient_task=st,
                            prob_of_ambiguous=args.prob_of_ambiguous,
                            togethercomputer=args.togethercomputer,
                            for_finetuning=True,
                            finetuning_control=args.finetuning_control,
                        )

                        all_tests = pd.concat([all_tests, curr_test], ignore_index=True)

        return all_tests

    def run_finetuned_set(self, args):
        """
        Generates finetuning set with two of the six features

        Args:
            args (ArgumentParser.args): command line arguments from main
        Returns:
            all_tests (pd.DataFrame): DataFrame containg the relevant information from all Prompts queried
        """
        all_tests = pd.DataFrame()

        construction_formats_list = [args.format_2, args.format_1]
        salient_tasks_list = ["propn", "negation"]

        construction_types_map = {
            "location": "subject_location",
            "subject": "subject_location",
            "religious": "religious_pronoun",
            "pronoun": "religious_pronoun",
            "propn": "propn_negation",
            "negation": "propn_negation",
        }

        all_tests = pd.DataFrame()
        for cf in construction_formats_list:
            for st in salient_tasks_list:
                for _ in range(
                    3
                ):  # need loop in order to not overload API and stay within OpenAI constraints
                    curr_test = self.run_test(
                        construction_type=construction_types_map[st],
                        shots=20,
                        model=args.model,
                        construction_format=cf,
                        crfm=args.crfm,
                        queries=20,
                        needs_instruction=True,
                        verbose=args.verbose,
                        needs_informative=False,
                        include_ambiguous_examples=True,
                        salient_task=st,
                        prob_of_ambiguous=args.prob_of_ambiguous,
                        togethercomputer=args.togethercomputer,
                        for_finetuning=False,
                        finetuning_control=False,
                    )

                    all_tests = pd.concat([all_tests, curr_test], ignore_index=True)
                    time.sleep(60)

        return all_tests
