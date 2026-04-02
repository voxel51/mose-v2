"""
MOSEv2 Dataset Loader for FiftyOne

MOSEv2 is a large-scale Video Object Segmentation benchmark. Archives are
distributed via Google Drive as tar.gz files of JPEG frame sequences and
indexed PNG annotation masks.

After extraction the layout is:
    dataset_dir/
        train/   (and/or valid/)
            JPEGImages/<sequence_name>/{00000,00001,...}.jpg
            Annotations/<sequence_name>/{00000,00001,...}.png

A symlink ``validation`` -> ``valid`` is created when the validation split
is present so FiftyOne's split directory checks match on-disk layout.

Annotation masks are 8-bit indexed PNGs where pixel value 0 is background
and pixel value N is object instance N.

Each FiftyOne sample represents one video frame. Samples carry:
    - sequence_id  (str)  — the video sequence name
    - frame_number (int)  — 0-based frame index
    - tags         [str]  — [split name, sequence_name]
    - ground_truth        — fo.Detections converted from the indexed PNG mask
                           (each Detection has label=str(obj_id-1), bounding_box,
                            mask, and index=obj_id-1)
"""

import os
from glob import glob
import tarfile

import cv2
import numpy as np
from PIL import Image

import fiftyone as fo

# FiftyOne split name: MOSEv2 split name as downloaded
SPLIT_TO_FOLDER = {
    "train": "train",
    "validation": "valid",
}

# Google Drive file IDs
DRIVE_FILE_IDS = {
    "train": "1o8Nd9t6oT_ZXmWHlImMzv5i8mn5vwRHm",
    "validation": "1iO8dScuVsGXLnrVggVT6C-wSGzN5tiCq",
}


def _drive_download_url(file_id):
    return f"https://drive.google.com/uc?id={file_id}"


def _ensure_symlink(dataset_dir, from_name, to_name):
    from_path = os.path.join(dataset_dir, from_name)
    to_path = os.path.join(dataset_dir, to_name)
    if not os.path.lexists(to_path):
        os.symlink(from_path, to_path)


def _count_frames(jpeg_dir):
    count = 0
    for seq in os.listdir(jpeg_dir):
        count += len(glob(os.path.join(jpeg_dir, seq, "*.jpg")))
    return count


def _segmentation_to_detections(
    segmentation: fo.Segmentation,
) -> fo.Detections:
    """Convert an indexed-PNG fo.Segmentation to fo.Detections.

    MOSE annotation masks are 8-bit indexed PNGs where pixel value = object
    instance ID (0 = background). This mirrors what the DAVIS loader produces
    natively, so the propagation operator sees the same input format.
    """
    mask = np.array(Image.open(segmentation.mask_path))  # type: ignore[arg-type]
    h, w = mask.shape

    detections = []
    for obj_id in np.unique(mask):
        if obj_id == 0:
            continue  # background
        binary = (mask == obj_id).astype(np.uint8)
        x, y, bw, bh = cv2.boundingRect(binary)
        if bw == 0 or bh == 0:
            continue
        detections.append(
            fo.Detection(
                label=str(obj_id - 1),
                bounding_box=[x / w, y / h, bw / w, bh / h],
                mask=binary[y : y + bh, x : x + bw],
                index=obj_id - 1,
            )
        )
    return fo.Detections(detections=detections)


def _load_image_dataset(dataset, split_dir, split_tag, max_samples=None):
    """Add all frames from a split directory to *dataset*.

    Args:
        dataset: FiftyOne dataset to populate
        split_dir: path to the extracted split folder (contains JPEGImages/ and Annotations/)
        split_tag: string tag to attach to every sample (e.g. "validation")
        max_samples (None): if set, stop after this many samples
    """
    jpeg_dir = os.path.join(split_dir, "JPEGImages")
    annot_dir = os.path.join(split_dir, "Annotations")

    sequences = sorted(os.listdir(jpeg_dir))

    samples = []
    for seq in sequences:
        frame_paths = sorted(glob(os.path.join(jpeg_dir, seq, "*.jpg")))
        for frame_path in frame_paths:
            if max_samples is not None and len(samples) >= max_samples:
                break
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
                sample["ground_truth"] = _segmentation_to_detections(
                    fo.Segmentation(mask_path=mask_path)
                )

            samples.append(sample)

    dataset.add_samples(samples)
    print(f"Added {len(samples)} samples.")


# ---------------------------------------------------------------------------
# Zoo interface — called by fiftyone.zoo.load_zoo_dataset
# ---------------------------------------------------------------------------


def download_and_prepare(dataset_dir, split="train", **kwargs):
    """Download and extract the requested split from the Google Drive source

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
        import gdown
    except ImportError as e:
        raise ImportError(
            "gdown is required to download MOSEv2 from Google Drive. "
            "Install it with: pip install gdown"
        ) from e

    if split is not None and split not in DRIVE_FILE_IDS:
        raise ValueError(
            f"Invalid split '{split}'. Supported splits: {list(DRIVE_FILE_IDS.keys())}"
        )

    os.makedirs(dataset_dir, exist_ok=True)
    splits_to_download = [split] if split else list(DRIVE_FILE_IDS.keys())

    total_frames = 0
    for split in splits_to_download:
        folder_name = SPLIT_TO_FOLDER[split]
        extract_dir = os.path.join(dataset_dir, folder_name)
        jpeg_dir = os.path.join(extract_dir, "JPEGImages")

        if not os.path.exists(jpeg_dir):
            tar_filename = f"{folder_name}.tar.gz"
            tar_path = os.path.join(dataset_dir, tar_filename)

            if os.path.isfile(tar_path) and os.path.getsize(tar_path) > 0:
                print(f"{tar_filename} already exists, skipping download")
            else:
                file_id = DRIVE_FILE_IDS[split]
                url = _drive_download_url(file_id)
                print(f"Downloading {tar_filename} from Google Drive...")
                gdown.download(url, tar_path, quiet=False, fuzzy=False)
                if (
                    not os.path.isfile(tar_path)
                    or os.path.getsize(tar_path) == 0
                ):
                    raise RuntimeError(
                        f"Download failed or empty file: {tar_path}"
                    )
                print(f"Downloaded {tar_filename}")

            print(f"Extracting {tar_filename}...")

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

        if folder_name != split:
            _ensure_symlink(dataset_dir, from_name=folder_name, to_name=split)

        total_frames += _count_frames(jpeg_dir)

    return None, total_frames, None


def load_dataset(dataset, dataset_dir, split=None, max_samples=None, **kwargs):
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
    for split in splits_to_load:
        folder_name = SPLIT_TO_FOLDER[split]
        split_dir = os.path.join(dataset_dir, folder_name)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}. "
                "Run download_and_prepare first."
            )
        _load_image_dataset(
            dataset, split_dir, split_tag=split, max_samples=max_samples
        )

    dataset.persistent = True
