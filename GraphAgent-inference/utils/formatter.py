import re

class CompactedStr(str):
    def apply_formatting(self):
        # remove successive spaces and newlines
        return re.sub(r'\n+', '\n', re.sub(r' +', ' ', self)).strip()