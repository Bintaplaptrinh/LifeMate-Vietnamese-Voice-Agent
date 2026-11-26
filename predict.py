import sys
from pathlib import Path


def main(argv=None):
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from lifemate.cli import main as run

    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
