"""
Generate a set of AmbiBench-style examples based on the given configuration.
Instead of running the inference pipeline, these examples are stored in a JSON file.

"""

import argparse
import datetime
import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Dict, List

from src.structures.api_access import OpenAI_APIAccess
from src.structures.prompt import Prompt

logger = logging.getLogger("GenerateDataset")
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S",
    level=logging.INFO,
)

CONSTRUCT_TYPE_MAP = {
    "location": "subject_location",
    "subject": "subject_location",
    "religious": "religious_pronoun",
    "pronoun": "religious_pronoun",
    "propn": "propn_negation",
    "negation": "propn_negation",
}

_CONSTRUCTION_TYPE_CHOICES = [
    "subject",
    "location",
    "religious",
    "negation",
    "propn",
    "pronoun",
]


@dataclass
class AmbiBenchConfig:

    construction_format: str
    n_shots: int
    n_queries: int
    prob_of_ambiguous: float

    needs_instruction: bool = False
    needs_informative: bool = False
    include_ambiguous_examples: bool = False
    construction_types: List[str] = field(
        default_factory=list,
        metadata={"help": "List of tasks or categories for which to generate examples"},
    )

    # model: str

    @classmethod
    def from_dict(cls, params):
        class_fields = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in params.items() if k in class_fields})


@dataclass
class AmbiBenchDataset:

    date: str
    config: AmbiBenchConfig
    examples: List[Dict[str, str]] = field(
        default_factory=list, metadata={"help": "List of query-completion tuple"}
    )

    @classmethod
    def from_dict(cls, params):
        class_fields = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in params.items() if k in class_fields})

    def __post_init__(self):
        if isinstance(self.config, dict):
            self.config = AmbiBenchConfig(**self.config)


class DatasetGenerator:
    def __init__(self, config: AmbiBenchConfig) -> None:
        self.config = config

        self.dataset = AmbiBenchDataset(
            date=datetime.datetime.now().strftime("%Y%m%d_%H-%M"), config=config
        )

    def generate_examples(self, n_queries):
        # adapted from `Tester.run_two_feature_tests_with_two_set`
        # two-feature tests {'subject_location', 'religious_pronoun', 'propn_negation'}

        # TODO: change the following depending on test cases
        # for now assume values for two-feature tests
        for_finetuning = True
        finetuning_control = False
        for salient_task in self.config.construction_types:

            if salient_task in CONSTRUCT_TYPE_MAP:
                construction_type = CONSTRUCT_TYPE_MAP[salient_task]
            else:
                logger.warning(
                    f"Salient task '{salient_task}' does not have valid mapping to construction type -> Skipped!"
                )
                continue

            for i in range(n_queries):

                prompt = Prompt(
                    shots=self.config.n_shots,
                    construction_type=construction_type,
                    format_type=self.config.construction_format,
                    needs_instruction=self.config.needs_instruction,
                    needs_informative=self.config.needs_informative,
                    include_ambiguous_examples=self.config.include_ambiguous_examples,
                    salient_task=salient_task,
                    prob_of_ambiguous=self.config.prob_of_ambiguous,
                    for_finetuning=for_finetuning,
                    finetuning_control=finetuning_control,
                )

                api_access = OpenAI_APIAccess(prompt)  # this object formats the prompt
                formatted_pair = api_access.generate_data_for_openai_finetuning(
                    format=self.config.construction_format,
                    needs_instruction=self.config.needs_instruction,
                )  # { "prompt": prompt, "completion": completion }

                formatted_pair["salient_task"] = prompt.examples[0].salient_task

                self.dataset.examples.append(formatted_pair)

    def save_examples_as_json(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(
            output_dir, f"{self.dataset.date}_ambibench_examples.json"
        )

        with open(file_path, "w", encoding="utf-8") as f_out:
            json.dump(asdict(self.dataset), f_out, indent=4)


def _get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--construction_types",
        nargs="+",
        required=False,
        default="location",
        help="Provide 1 or more tasks/categories for which to generate examples",
    )  # choices=_CONSTRUCTION_TYPE_CHOICES,
    parser.add_argument(
        "--construction_format",
        choices=["arrow", "qa"],
        type=str,
        required=False,
        default="qa",
    )
    parser.add_argument(
        "--n_shots",
        type=int,
        required=False,
        default=1,
        help="Number of shots per query",
    )
    parser.add_argument(
        "--n_queries",
        type=int,
        required=False,
        default=10,
        help="Number of queries/examples to generate",
    )
    # parser.add_argument('--model', type=str, required=False, default="text-davinci-003")
    parser.add_argument("--needs_instruction", action="store_true")
    parser.add_argument("--needs_informative", action="store_true")
    parser.add_argument("--needs_multiple_choice", action="store_true")
    parser.add_argument("--include_ambiguous_examples", action="store_true")
    parser.add_argument("--verbose", type=bool, required=False, default=True)
    parser.add_argument("--prob_of_ambiguous", type=float, required=False, default=50)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _get_args()

    config = AmbiBenchConfig.from_dict(vars(args))

    data_generator = DatasetGenerator(config)
    data_generator.generate_examples(config.n_queries)
    data_generator.save_examples_as_json(output_dir="./for_finetuning")
