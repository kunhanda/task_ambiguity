import pandas as pd

from src.structures.api_access import OpenAI_APIAccess
from src.structures.metric_wrangler import MetricWrangler
from src.structures.prompt import Prompt


class QueryPipeline:
    """
    To test generating prompts, and querying the API, and parsing the output

    Attributes:
        construction_type (str): the type of examples to generate: one of {subject_location, religious_pronoun, propn_negation}
        shots (int): the number of shots to query the API with for each prompt
        model (str): the OpenAI model to query with the prompts
        construction_format (str): format of examples to generate: one of {qa, arrow}
        crfm (bool): True if running tests on Stanford CRFM, False otherwise
    """

    def __init__(self, construction_type, shots, model, construction_format, crfm):
        self.construction_type = construction_type
        self.shots = shots
        self.model = model
        self.construction_format = construction_format
        self.crfm = crfm

    def run_pipeline(
        self,
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
        Creates a sample test pipeline with which to generate prompts, query the API, and parse the output
        Args:
            queries (str): the number of prompts to query the API with
            needs_instruction (bool): True if Prompt requires instruction, False otherwise
            verbose (bool): True to provide printed output, False otherwise
            needs_informative (bool): True if requires informative instruction, False otherwise
            include_ambiguous_examaples (bool): True if wish to include ambiguous examples and False otherwise
            prob_of_ambiguous (float): Number from 0.0 to 1.0 indicating the probability of each example generated being an ambigous example
            togethercomputer (bool): True if for t0pp example generation, False otherwise
            for_finetuning (bool): True if generating examples with withheld salient tasks for finetuning
            finetuning_control (bool): True if running tests for finetuning control and False otherwise
            salient_task (str): salient task for which to make examples (not required to generate examples)

        Returns:
            complete_test_df (pd.DataFrame): a DataFrame containing all of the information
                from the set of Prompts for the current construction_type + format_type
        """
        wrangler = MetricWrangler()
        examples = []
        test_examples_output_df = pd.DataFrame()

        for i in range(queries):
            prompt = Prompt(
                construction_type=self.construction_type,
                shots=self.shots,
                format_type=self.construction_format,
                needs_instruction=needs_instruction,
                needs_informative=needs_informative,
                include_ambiguous_examples=include_ambiguous_examples,
                salient_task=salient_task,
                prob_of_ambiguous=prob_of_ambiguous,
                for_finetuning=for_finetuning,
                finetuning_control=finetuning_control,
            )

            examples.extend(prompt.get_examples())

            if verbose:
                prompt.print()

            api_access = OpenAI_APIAccess(prompt)

            if for_finetuning:
                formatted_pair = api_access.generate_data_for_openai_finetuning(
                    format=self.construction_format, needs_instruction=needs_instruction
                )
                # optional
                api_access.store_prompt_completion_pair_as_jsonl(formatted_pair)

            elif togethercomputer:
                if self.construction_format == "qa":
                    max_tokens = 2
                else:
                    max_tokens = 1
                api_access.to_togethercomputer(
                    format=self.construction_format,
                    request_type="language-model-inference",
                    model="t0pp",
                    needs_instruction=needs_instruction,
                    max_tokens=max_tokens,
                    logprobs=4,
                )
            else:
                output = api_access.request(
                    self.model, self.construction_format, needs_instruction
                )
                unpacked_df = api_access.to_numpy_dataframe(output)

                probs_df = api_access.isolate_probs(unpacked_df)

                if verbose:
                    print("CURRENT PROMPT PROBS DF")

                labeled_df = wrangler.label_probs(probs_df, needs_instruction)

                if verbose:
                    print(labeled_df)

                test_examples_output_df = test_examples_output_df.append(
                    labeled_df, ignore_index=True
                )
                if verbose:
                    print(test_examples_output_df)

        complete_test_df = wrangler.construct_test_example_df(
            test_examples=examples, test_examples_output_df=test_examples_output_df
        )

        print("TEST EXAMPLES DF\n" + str(complete_test_df))

        return complete_test_df
