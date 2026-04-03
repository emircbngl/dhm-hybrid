import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    src_dir = root / "src"
    sys.path.insert(0, str(src_dir))
    import main as app_main  # type: ignore

    app_main.main()


if __name__ == "__main__":
    main()
