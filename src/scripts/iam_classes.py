class WordLineData:
    def __init__(self, line):
        """Constructor."""
        self._line = line
        self.valid = False
        self.path = ""
        self.ok = True
        self.gray = 0
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0
        self.grammar = ""
        self.word = ""

        self.valid = self._parse_line()

    def _parse_line(self):
        """Parses the line."""
        parts = self._line.split(" ")
        try:
            self.path = parts[0]
            self.ok = parts[1] == "ok"
            self.gray = int(parts[2])
            self.x = int(parts[3])
            self.y = int(parts[4])
            self.w = int(parts[5])
            self.h = int(parts[6])
            self.grammar = parts[7]
            self.word = " ".join(parts[8:])
        except ValueError as e:
            return False

        return True