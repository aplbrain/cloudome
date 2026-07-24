from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cloudome.manage import *  # noqa: F401,F403
from cloudome.manage import main


if __name__ == "__main__":
    main()
