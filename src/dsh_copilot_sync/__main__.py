"""Package entry point: ``python -m dsh_copilot_sync``."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
