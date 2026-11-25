"""Download the dataset archive linked by the model authors; keep it local."""

from io import BytesIO
import argparse
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen
from zipfile import ZipFile

URL = "https://www.researchgate.net/profile/Pekka_Malo/publication/251231364_FinancialPhraseBank-v10/data/0c96051eee4fb1d56e000000/FinancialPhraseBank-v10.zip?origin=publication_list"


def main() -> None:
    """Extract the all-agreement file into the evaluator's expected location."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, help="Use a manually downloaded archive"
    )
    args = parser.parse_args()
    if args.archive:
        payload = args.archive.read_bytes()
    else:
        try:
            with urlopen(URL, timeout=60) as response:
                payload = response.read()
        except HTTPError as error:
            raise SystemExit(
                f"Provider returned HTTP {error.code}. Download the archive in your browser and rerun with --archive PATH."
            ) from error
    with ZipFile(BytesIO(payload)) as archive:
        names = [
            name
            for name in archive.namelist()
            if Path(name).name == "Sentences_AllAgree.txt"
        ]
        if len(names) != 1:
            raise SystemExit(
                "Expected exactly one Sentences_AllAgree.txt in the archive"
            )
        destination = (
            root
            / "data/FinancialPhraseBank-v1.0/FinancialPhraseBank-v1.0/Sentences_AllAgree.txt"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(archive.read(names[0]))
    print("Saved", destination)


if __name__ == "__main__":
    main()
