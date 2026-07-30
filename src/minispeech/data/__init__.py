from minispeech.data.alignment import AlignLevel, monotonic_gaussian_prior
from minispeech.data.dataset import MiniSpeechDataset, collate_minispeech, make_stop_target
from minispeech.data.manifest import Utterance, read_manifest, write_manifest
from minispeech.data.mel import MelCacheReader, MelCacheWriter, waveform_to_logmel
from minispeech.data.shards import ShardReader, ShardWriter

__all__ = [
    "AlignLevel",
    "MelCacheReader",
    "MelCacheWriter",
    "MelSpeechDataset",
    "MiniSpeechDataset",
    "ShardReader",
    "ShardWriter",
    "Utterance",
    "collate_mel",
    "collate_minispeech",
    "make_stop_target",
    "monotonic_gaussian_prior",
    "read_manifest",
    "waveform_to_logmel",
    "write_manifest",
]


def __getattr__(name: str):
    if name in {"MelSpeechDataset", "collate_mel"}:
        from minispeech.data.mel_dataset import MelSpeechDataset, collate_mel

        return MelSpeechDataset if name == "MelSpeechDataset" else collate_mel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")