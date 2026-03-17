"""Dataset sampling and batching."""

from __future__ import annotations

from torch.utils.data import DataLoader, Dataset

from src.data.types import Prompt


class PromptDataset(Dataset):
    """Thin wrapper around a list of Prompts for use with DataLoader."""

    def __init__(self, prompts: list[Prompt]) -> None:
        self._prompts = prompts

    def __len__(self) -> int:
        return len(self._prompts)

    def __getitem__(self, index: int) -> Prompt:
        return self._prompts[index]


def _collate_prompts(batch: list[Prompt]) -> list[Prompt]:
    """Identity collation — return batch as-is.

    Prompt token_ids are variable-length, so padding is deferred to the
    training loop where the target length is known.
    """
    return batch


def create_dataloader(
    prompts: list[Prompt],
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader yielding batches of Prompts."""
    dataset = PromptDataset(prompts)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate_prompts,
    )
