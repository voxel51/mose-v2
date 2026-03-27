# mose-v2

<div align="center">

[![Discord](https://img.shields.io/badge/Discord-7289DA?logo=discord&logoColor=white)](https://discord.gg/fiftyone-community)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-purple?style=flat&logo=huggingface)](https://huggingface.co/Voxel51)
[![Voxel51 Blog](https://img.shields.io/badge/Voxel51_Blog-ff6d04?style=flat)](https://voxel51.com/blog)
[![Newsletter](https://img.shields.io/badge/Newsletter-BE5B25?logo=mail.ru&logoColor=white)](https://share.hsforms.com/1zpJ60ggaQtOoVeBqIZdaaA2ykyk)
[![LinkedIn](https://img.shields.io/badge/In-white?style=flat&label=Linked&labelColor=blue)](https://www.linkedin.com/company/voxel51)
[![Twitter](https://img.shields.io/badge/Twitter-000000?logo=x&logoColor=white)](https://x.com/voxel51)
[![Medium](https://img.shields.io/badge/Medium-12100E?logo=medium&logoColor=white)](https://medium.com/voxel51)

</div>

A [FiftyOne remote zoo dataset](https://docs.voxel51.com/dataset_zoo/remote.html) integration for **MOSEv2**, a large-scale video object segmentation benchmark: thousands of videos, instance masks, and diverse real-world conditions (occlusion, small objects, weather, low light, camouflage, etc.). See the [project site](https://mose.video/) and [upstream repo](https://github.com/FudanCVL/MOSEv2) for the full benchmark description.


### Source and citation

- **Website**: [mose.video](https://mose.video/)
- **GitHub**: [MOSEv2](https://github.com/FudanCVL/MOSEv2)
- **Hugging Face (dataset card)**: [FudanCVL/MOSEv2](https://huggingface.co/datasets/FudanCVL/MOSEv2)
- **License**: Original MOVEv2 terms: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
- **Citation**: See also other related citations from the [official MOSEv2 website](https://mose.video/#citation).
```
@article{MOSEv2,
  title={{MOSEv2}: A More Challenging Dataset for Video Object Segmentation in Complex Scenes},
  author={Ding, Henghui and Ying, Kaining and Liu, Chang and He, Shuting and Jiang, Xudong and Jiang, Yu-Gang and Torr, Philip HS and Bai, Song},
  journal={arXiv preprint arXiv:2508.05630},
  year={2025}
}
```

## Quick start

Installation

```bash
pip install fiftyone
pip install gdown   # required for Google Drive download; see also requirements.txt
```

Load via the FiftyOne Dataset Zoo

```python
import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset(
    "https://github.com/voxel51/mose-v2",
    split="train",  # or "validation"
    max_samples=1000,  # optional, for quicker exploration
)

session = fo.launch_app(dataset)

# For a dynamic Grouped view
grouped_view = dataset.group_by("sequence_id", order_by="frame_number")
```

## Notes:

- Downloads **train** and **validation** archives from Google Drive (file IDs are in `__init__.py` as `DRIVE_FILE_IDS`).
- Extracts `train/` and `valid/` under the FiftyOne-managed dataset directory. A symlink `validation` → `valid` is created when needed so split names match FiftyOne’s expectations.
```text
dataset_dir/
  train/
    JPEGImages/<sequence_name>/{00000,00001,...}.jpg
    Annotations/<sequence_name>/{00000,00001,...}.png
  valid/
    JPEGImages/<sequence_name>/{00000,00001,...}.jpg
    Annotations/<sequence_name>/00000.png
```
- Registers **one sample per video frame**. Segmentation is stored as an indexed PNG per frame (`ground_truth`: `fo.Segmentation` with `mask_path`).
- Annotation masks are **8-bit indexed PNGs**: pixel value `0` is background; value `N` is object instance `N`.

## Sample fields

| Field | Role |
|-------|------|
| `filepath` | Path to the JPEG frame |
| `sequence_id` | Video sequence name |
| `frame_number` | Zero-based frame index |
| `tags` | Split and sequence (e.g. `train`, sequence id) |
| `ground_truth` | `Segmentation` with `mask_path` to the indexed PNG |


## Statistics

| Split       | Sequences | Total Samples | Annotated Samples      |
|-------------|-----------|---------------|------------------------|
| train       | 3,666     | 311,843       | 311,843                |
| validation  | 433       | 66,526        | 433 (first frame only) |


## Visualize

Each image is tagged with its **split** and with its **sequence** name — frames that share a `sequence_id` belong to the same clip.

For a **video-like** browser in the App, use a dynamic grouped view — one group per sequence, frames ordered by `frame_number`.

![MOSEv2 sample visualization (grid)](assets/mose-grid.png)

![MOSEv2 grouped / carousel view](assets/mose-carousel.png)
