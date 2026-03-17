"""Download and cache SQuAD 2.0 and MeetingBank datasets."""

from datasets import Dataset, load_dataset


def _print_split_stats(ds_split: Dataset, name: str, text_key: str) -> None:
    if text_key not in ds_split.column_names:
        raise ValueError(f"Key {text_key!r} not found in dataset columns: {ds_split.column_names}")
    n = len(ds_split)
    avg_len = sum(len(ex[text_key]) for ex in ds_split) / max(n, 1)
    print(f"  {name}: {n} examples, avg {avg_len:.0f} chars")


def download_squad() -> None:
    print("Downloading SQuAD 2.0...")
    ds = load_dataset("squad_v2")
    for split_name, split_ds in ds.items():
        _print_split_stats(split_ds, split_name, "context")


def download_meetingbank() -> None:
    print("Downloading MeetingBank...")
    ds = load_dataset("lytang/MeetingBank-transcript")
    for split_name, split_ds in ds.items():
        _print_split_stats(split_ds, split_name, "source")


def download_spacy_model(model_name: str = "en_core_web_sm") -> None:
    """Download spaCy language model for Selective Context baseline."""
    import subprocess
    import sys

    print(f"Downloading spaCy model '{model_name}'...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])


def main() -> None:
    download_squad()
    download_meetingbank()
    download_spacy_model()
    print("Done.")


if __name__ == "__main__":
    main()
