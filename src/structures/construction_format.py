class ConstructionFormat:
    """
    Creates a stencil format for the prompt
    """

    def make_prefix(self):
        pass

    def make_infix(self):
        pass

    def make_suffix(self):
        pass

    def get_affixes(self):
        return (self.make_prefix(), self.make_infix(), self.make_suffix())


class ArrowFormat(ConstructionFormat):
    """
    Creates the 'arrow' (>) stencil format.
    """

    def make_prefix(self):
        return ""

    def make_infix(self):
        return "\n>"

    def make_suffix(self):
        return ""


class QAFormat(ConstructionFormat):
    """
    Creates the 'qa' (Q: ,A: ) stencil format.
    """

    def make_prefix(self):
        return "Q: "

    def make_infix(self):
        return "\nA: "

    def make_suffix(self):
        return ""
