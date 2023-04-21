from dataclasses import dataclass


@dataclass
class Example:
    """
    A dataclass to create an example construction and retain relevant information on the two possible tasks within that construction
    An Example is a Construction + associated metadata regarding that Example

    For example:
    construction_type: 'subject_location'  # construction_type ['subject_location', 'religious_pronoun', 'propn_negation',
    'subject', 'location', 'religious', 'pronoun', 'propn', 'negation']
    construction: 'The critic is in the theatre'  # the construction (sentence) itself.
    task_a_label: True  # the label for the first task, in this case, if task_a_label is True,
        the contstruction contains a reference to a human (as opposed to an animal)
    task_b_label: True  # the label for the second task, in this case, if task_b_label is True,
        the contstruction contains a reference to an indoor location (as opposed to an outdoor location)
    active_task_label: True  # the label for the example overall -- the "user-facing" label for the example
    salient_task: 'task_a'  # the task that determines the salient task -- the task that determines the active_task_label

    Attributes:x
        construction_type (str): the type of construction contained in this example
        construction (str): the construction text
        task_a_label (bool): the label of the first task
        task_b_label (bool): the label of the second task
        active_task_label (bool): label of the overall construction,
            the active task [task_a, task_b] is the task that determines the overall label of the construction
        salient_task (str): the name of the task that is the salient task for this construction, determined after instantiation
    """

    construction_type: str
    format_type: str
    construction: str
    task_a_label: bool
    task_b_label: bool
    active_task_label: bool
    salient_task: str = None

    def as_dict(self):
        return {
            "construction_type": self.construction_type,
            "salient_task": self.salient_task,
            "format_type": self.format_type,
            "construction": self.construction,
            "task_a_label": self.task_a_label,
            "task_b_label": self.task_b_label,
            "active_task_label": self.active_task_label,
        }
