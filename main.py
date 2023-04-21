import argparse

from src.structures.construction_types import ConstructionType
from src.tester import Tester

_CONSTRUCTION_TYPE_CHOICES = ConstructionType.list()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type_1",
        choices=_CONSTRUCTION_TYPE_CHOICES,
        type=str,
        required=False,
        default="subject_location",
    )
    parser.add_argument(
        "--type_2",
        choices=_CONSTRUCTION_TYPE_CHOICES,
        type=str,
        required=False,
        default="religious_pronoun",
    )
    parser.add_argument(
        "--type_3",
        choices=_CONSTRUCTION_TYPE_CHOICES,
        type=str,
        required=False,
        default="propn_negation",
    )
    parser.add_argument("--shots", type=int, required=False, default=1)
    parser.add_argument("--model", type=str, required=False, default="text-davinci-003")
    parser.add_argument(
        "--format_1", choices=["arrow", "qa"], type=str, required=False, default="arrow"
    )
    parser.add_argument(
        "--format_2", choices=["arrow", "qa"], type=str, required=False, default="qa"
    )
    parser.add_argument("--needs_instruction", type=bool, required=False, default=True)
    parser.add_argument("--needs_informative", type=bool, required=False, default=False)
    parser.add_argument(
        "--include_ambiguous_examples", type=bool, required=False, default=True
    )
    parser.add_argument("--verbose", type=bool, required=False, default=True)
    parser.add_argument("--crfm", type=bool, required=False, default=False)
    parser.add_argument("--prob_of_ambiguous", type=float, required=False, default=50)
    parser.add_argument("--togethercomputer", type=bool, required=False, default=False)
    parser.add_argument(
        "--finetuning_control", type=bool, required=False, default=False
    )

    args = parser.parse_args()
    tester = Tester()

    all_tests = tester.run_two_feature_tests_with_two_set(args)
    # all_tests = tester.run_baseline_tests_for_finetuning(args)
    file_name = "finetune_test"
    all_tests.to_csv(file_name)


if __name__ == "__main__":
    main()
