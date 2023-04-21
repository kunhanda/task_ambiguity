from typing import Union

from src.example_generation import (
    ExampleGenerator,
    ProperNounNegationGenerator,
    ReligiousPronounGenerator,
    SubjectLocationGenerator,
)
from src.structures.common import ExtendedEnum


class ConstructionType(ExtendedEnum):

    SUBJECT_LOCATION = "subject_location"
    PROPN_NEGATION = "propn_negation"
    RELIGIOUS_PRONOUN = "religious_pronoun"
    LOCATION = "location"
    SUBJECT = "subject"
    NEGATION = "negation"
    PRONOUN = "pronoun"
    RELIGIOUS = "religious"
    PROPN = "propn"


def get_generator_from_construction_type(
    construction_type: Union[str, ConstructionType], format_type: str
) -> ExampleGenerator:

    if isinstance(construction_type, str):
        construction_type = ConstructionType(construction_type)

    if construction_type in [
        ConstructionType.LOCATION,
        ConstructionType.SUBJECT,
        ConstructionType.SUBJECT_LOCATION,
    ]:
        return SubjectLocationGenerator
    elif construction_type in [
        ConstructionType.PROPN,
        ConstructionType.NEGATION,
        ConstructionType.PROPN_NEGATION,
    ]:
        return ProperNounNegationGenerator
    elif construction_type in [
        ConstructionType.RELIGIOUS,
        ConstructionType.PRONOUN,
        ConstructionType.RELIGIOUS_PRONOUN,
    ]:
        return ReligiousPronounGenerator

    else:
        raise ValueError(
            f"Undefined mapping to generator for: {repr(construction_type)}"
        )
