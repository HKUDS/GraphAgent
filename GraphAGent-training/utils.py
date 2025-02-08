import sys
import pdb

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

import re

class CompactedStr(str):
    def apply_formatting(self):
        # remove successive spaces and newlines
        return re.sub(r'\n+', '\n', re.sub(r' +', ' ', self)).strip()