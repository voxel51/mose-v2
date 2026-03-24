"""
MOSEv2 Dataset Loader for FiftyOne

MOSEv2 (https://huggingface.co/datasets/FudanCVL/MOSEv2) is a large-scale
Video Object Segmentation benchmark. The dataset lives on HuggingFace as
tar.gz archives of JPEG frame sequences and indexed PNG annotation masks.

After extraction the layout is:
    dataset_dir/
        valid/
            JPEGImages/<sequence_name>/{00000,00001,...}.jpg
            Annotations/<sequence_name>/{00000,00001,...}.png

Annotation masks are 8-bit indexed PNGs where pixel value 0 is background
and pixel value N is object instance N.

Each FiftyOne sample represents one video frame. Samples carry:
    - sequence_id  (str)  — the video sequence name
    - frame_number (int)  — 0-based frame index
    - tags         [str]  — ["validation", <sequence_name>]
    - ground_truth        — fo.Segmentation with mask_path pointing to the PNG
"""

import os
from glob import glob

import fiftyone as fo


REPO_ID = "FudanCVL/MOSEv2"

# FiftyOne split name  ->  folder name inside the tar / on disk
SPLIT_TO_FOLDER = {
    "train": "train",
    "validation": "valid",
}


# ---------------------------------------------------------------------------
# Zoo interface — called by fiftyone.zoo.load_zoo_dataset
# ---------------------------------------------------------------------------


def download_and_prepare(dataset_dir, split=None, **kwargs):
    """Download and extract the requested split from HuggingFace Hub.

    Args:
        dataset_dir: directory managed by FiftyOne where data will be stored
        split (None): split to download; ``"train"`` or ``"validation"``.
            Pass ``None`` to download all available splits.

    Returns:
        (dataset_type, num_samples, classes) — dataset_type is always None
        (signals that ``load_dataset`` drives loading), num_samples is the
        total number of frames across all downloaded sequences, and classes
        is None.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to download MOSEv2. "
            "Install it with: pip install huggingface_hub"
        )

    if split is not None and split not in SPLIT_TO_FOLDER:
        raise ValueError(
            f"Invalid split '{split}'. Supported splits: {list(SPLIT_TO_FOLDER)}"
        )

    os.makedirs(dataset_dir, exist_ok=True)
    splits_to_download = [split] if split else list(SPLIT_TO_FOLDER)

    total_frames = 0
    for s in splits_to_download:
        folder_name = SPLIT_TO_FOLDER[s]
        extract_dir = os.path.join(dataset_dir, folder_name)
        jpeg_dir = os.path.join(extract_dir, "JPEGImages")

        if not os.path.exists(jpeg_dir):
            tar_filename = f"{folder_name}.tar.gz"
            tar_path = os.path.join(dataset_dir, tar_filename)

            if not os.path.exists(tar_path):
                print(f"Downloading {REPO_ID}/{tar_filename} from HuggingFace Hub...")
                hf_hub_download(
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    filename=tar_filename,
                    local_dir=dataset_dir,
                )

            print(f"Extracting {tar_filename}...")
            import tarfile
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(dataset_dir)

            if not os.path.exists(jpeg_dir):
                raise RuntimeError(
                    f"Extraction finished but expected directory not found: {jpeg_dir}. "
                    "The archive may have an unexpected layout."
                )
            print(f"Extraction complete: {extract_dir}")
        else:
            print(f"Found existing data at {extract_dir}, skipping download.")

        total_frames += _count_frames(jpeg_dir)

    return None, total_frames, None


def load_dataset(dataset, dataset_dir, split=None, **kwargs):
    """Load the dataset into the given FiftyOne dataset.

    Each video frame becomes one :class:`fiftyone.core.sample.Sample`.
    Samples are tagged with the split name and their sequence name so they
    can be filtered with :meth:`Dataset.match_tags`, grouped with
    ``dataset.group_by("sequence_id", order_by="frame_number")``, etc.

    Args:
        dataset: :class:`fiftyone.core.dataset.Dataset` to populate
        dataset_dir: directory where the data was downloaded
        split (None): split to load; ``"train"`` or ``"validation"``.
            Pass ``None`` to load all available splits.
    """
    if split is not None and split not in SPLIT_TO_FOLDER:
        raise ValueError(
            f"Invalid split '{split}'. Supported splits: {list(SPLIT_TO_FOLDER)}"
        )

    splits_to_load = [split] if split else list(SPLIT_TO_FOLDER)
    for s in splits_to_load:
        folder_name = SPLIT_TO_FOLDER[s]
        split_dir = os.path.join(dataset_dir, folder_name)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}. "
                "Run download_and_prepare first."
            )
        _load_image_dataset(dataset, split_dir, split_tag=s)

    dataset.persistent = True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _count_frames(jpeg_dir):
    count = 0
    for seq in os.listdir(jpeg_dir):
        count += len(glob(os.path.join(jpeg_dir, seq, "*.jpg")))
    return count


def _load_image_dataset(dataset, split_dir, split_tag):
    """Add all frames from a split directory to *dataset*.

    Args:
        dataset: FiftyOne dataset to populate
        split_dir: path to the extracted split folder (contains JPEGImages/ and Annotations/)
        split_tag: string tag to attach to every sample (e.g. "validation")
    """
    jpeg_dir = os.path.join(split_dir, "JPEGImages")
    annot_dir = os.path.join(split_dir, "Annotations")

    sequences = sorted(os.listdir(jpeg_dir))
    print(f"Loading {len(sequences)} sequences from {split_dir}...")

    samples = []
    for seq in sequences:
        frame_paths = sorted(glob(os.path.join(jpeg_dir, seq, "*.jpg")))
        for frame_path in frame_paths:
            stem = os.path.splitext(os.path.basename(frame_path))[0]
            frame_number = int(stem)
            mask_path = os.path.join(annot_dir, seq, f"{stem}.png")

            sample = fo.Sample(
                filepath=frame_path,
                tags=[split_tag, seq],
                sequence_id=seq,
                frame_number=frame_number,
            )

            if os.path.exists(mask_path):
                # Indexed PNG: pixel value = object instance ID (0 = background)
                sample["ground_truth"] = fo.Segmentation(mask_path=mask_path)

            samples.append(sample)

    dataset.add_samples(samples)
    print(f"Added {len(samples)} samples.")
