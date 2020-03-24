# Annotations for Multi Camera Multi Tracking

## Structure

```
mcmt_extra
├── DatasetName_0
│   ├── meta.json
│   ├── set_of_xml_files.zip
│   ├── single_annotation_0.xml
│   └── single_annotation_1.xml
│
├── DatasetName_1
│   ├── meta.json
│   ├── set_of_xml_files.zip
│   ├── single_annotation_0.xml
│   └── single_annotation_1.xml
...
├── scripts
├── .gitignore
├── README.md
├── __init__.py
└── main.py
```

Every directory with annotations may consists of *.zip archives with annotations
inside or individual *.xml annotations. Every directory must have meta-file
`meta.json` with configuration for every available annotation.


> *Note: directories started with `.` will be ignored*


## Meta-file

Example of meta-file:
```
{
    'name': "2892_DSS_ReIdentification_1_Airport_24mm_500_Lux.mp4.xml",
    'location': "/path/to/videos/Airport/1_Airport_24mm_500_Lux.mp4",
    'frame_rate': 1,
    'filter': {'frames': 30, 'occlusion': ['heavy_occluded'], 'view': ['__undefined__'], 'min_square': 3012},
    'data_type': 'video',
    'zip_archive': 'Reidentification_airport.zip',
    'scene': 0,
    'cam_id': 0,
    'target': 'test',
    'ignore': False,
},
{
    'name': "2893_DSS_ReIdentification_2_Airport_35mm_500_Lux.mp4.xml",
    'location': "/path/to/videos/Airport/2_Airport_35mm_500_Lux.mp4",
    'frame_rate': 1,
    'filter': {'frames': 30, 'occlusion': ['heavy_occluded'], 'view': ['__undefined__'], 'min_square': 3012},
    'data_type': 'video',
    'zip_archive': 'Reidentification_airport.zip',
    'scene': 0,
    'cam_id': 1,
    'target': 'test',
    'ignore': False,
}
```

Meta-file includes the next information:
* 'name' - name of file with annotation in `*.xml` format
* 'location' - full path to video file or directory with images
* 'frame_rate' - used for video according to a frame rate parameter in `CVAT` task (get every *N* frame from video)
* 'filter' - set of parameters which should be filtered:
  * 'frames' - process only every *N* frame (over `frame_rate`)
  * 'occlusion' - ignore objects with these values of occlusion
  * 'view' - ignore objects with these values of view (or orientation)
  * 'min_square' - ignore objects with square of bounding boxes less then this value
* 'data_type' - data type: `video` or `images`
* 'zip_archive' - if `*.xml` file is inside a zip archive, here is name of the `*.zip` file, `None` if not
* 'scene' - number of scene in the annotation
* 'cam_id' - ID of camera for the annotation
* 'target' - type of subset: `train`, `val` or `test`
* 'ignore' - may be `True` or `False`. If `True`, the annotation will be ignored
* 'tracks_comparison' - `*.xml` file with information about matching tracks
(recommended to save in zip archive)

> *Note: if directory with annotation does not have meta.json file, the annotation will be ignored*


## Setup

### Prerequisites

* Ubuntu 16.04
* Python 3

### Installation

1. Clone the `mcmt_extra` repository
2. Install python modules:
```bash
cd mcmt_extra
pip install -r scripts/requirements.txt
```

## Using

### Parsing annotation

This repository supports the following functions:
1. Create output annotation from available tracks in *.xml annotations
2. Create task for matching tracks
3. Patching *.xml annotations

Common steps of parsing annotation:
1. extract and save bounding boxes from videos/images
2. filter available bounding boxes
3. Create output annotation as txt files

To show all available parameters of the script use the next command:
```bash
python scripts/main.py -h
```

To parse all annotations use the next command:
```bash
python scripts/main.py --output_dir /path/to/output/dir
```

Argument `output_dir` id required.

If you want to parse only chosen annotation, use parameter `--parse_only`. For example:
```bash
python scripts/main.py --output_dir /path/to/output/dir --parse_only Mall Airport
```

If input annotations consist of "raw" tracks which need matching should be used
argument `--split_dirs_by_source`. Tracks in every annotation have serial number
begins from zero and these numbers can be the same in different annotations. To
avoid unexpected merging of tracks should be used the argument.

It is possible to extract frames from videos. For this use parameter `--extract_frames`.
After extracting frames the script finishes work.

During execution the script filters samples for every identity. Use parameter
`--target_samples_num` to set up the maximum number of samples per identity.
By default is equal to 64.
Finally not all extracted bounding boxes are in output annotation. You can save
images with bounding boxes from the output annotation,
use parameter `--save_filtered_bboxes`.

Extracting bounding boxes from videos/images is a quite long process.
If you already extracted all needed bounding boxes, you can avoid this process
and work with available images, key: `--skip_extracting_bboxes`.

For extracting bounding boxes available option `--expand_bboxes` that
expands height and width of bounding boxes from its center.
Default value equal to (1.0, 1.0) and means do not change size.

The result annotation is saved to several files (in directory `--output_dir`):
* `train.txt`
* `test_gallery.txt`
* `test_query.txt`
* `val_gallery.txt`
* `val_query.txt`

Every txt file consists of image list with related path. The script
after every running appends image list to an appropriate txt file.
If you want to rewrite these txt files, use key `--rewrite_output_txt_files`.

Test subset is splitted onto two ones: `query` and `gallery`. With parameter
`--query_part` can be set up percentage of query part in test subset.
By default is equal to 0.2.

Minimum number of samples per identity set up with parameter `--min_track_len`.
By default is equal to 2.

The result annotation can be saved in format of `Market1501`. For this use
parameter `--market1501_format`.

Finally the script collects statistics from output txt files. It is possible
to collect statistics without parsing annotations via parameter `--show_statistics_only`.
Example of output statistics:
```
  TRAIN:
    identities: 31
    samples number: 1118
  TEST:
    query:
      identities: 777
      samples number: 4176
    gallery:
      identities: 777
      samples number: 16218
  VAL:
    query:
      identities: 0
      samples number: 0
    gallery:
      identities: 0
      samples number: 0
```

### Search for similar tracks

This mode uses ReId model with `OpenVINO` to find the closest tracks by
cosine distance in annotations. The result is saved to images. Every image
consists of bounding boxes belonging to two different tracks. To enable this mode
use the key `--find_similar_tracks`.

Pipeline:
1. extract and save bounding boxes from videos/images
2. get `--samples_proc_num` number of samples per track and extract embeddings
3. calculate distance between all samples
4. find topK closest tracks
5. create and save output images

Firstly you need to have installed `OpenVINO` and ReId model in IR format (*.xml and *.bin)
files. With parameter `--reid_model` choose *.xml file of ReId model (*.bin must be
in the same directory). In some cases you should set up path to cpu extension for
`OpenVINO`. Use parameter `--cpu_extension`. For more information see documentation
on `OpenVINO`.

The next parameters are option but can be changed:
* `--samples_proc_num` - maximum number of samples per track for extracting embeddings, default: 32
* `--topk_closest_tracks` - maximum number of the closest tracks to find, default: 5
* `--samples_vis_num` - maximum number of samples per track which are drawing on output images, default: 16
* `--sample_size` - size of samples which are drawing on output images: default 128x160

### Patching annotations

"Raw" tracks can not be used in final annotation. To resolve this situation
we can firstly use mode `searching for similar tracks` to create a task for annotators
and use this new annotations with compared and matched tracks for patching
our original *.xml annotations with new ID of tracks. To enable this mode use
key `--patch_xml`. In this mode will be updated annotations that have field
`tracks_comparison` in meta-file. For every scene will be merged tracks
that marked as "the same" in annotation from `tracks_comparison` entry.
From zip archives extracted original annotations in `original_annotations`
directory, patched annotations will be saved to `patched_annotations` directory.
In zip archie or non-zipped original annotations must be replaced by patched ones
manually. With that must be updated meta-file (new name of annotations) if it needs.

### Example of using the whole process
1.  Suppose we have two annotations with tracks and belong to scene 0:
    * `Anno#1.xml`
    * `Anno#2.xml`
2. The structure of our repository looks like the next:
```
mcmt_extra
├── Anno
│   ├── meta.json
│   └── Anno.zip
│       ├── Anno#1.xml
│       └── Anno#2.xml
├── scripts
├── .gitignore
├── README.md
├── __init__.py
└── main.py
```
3.  Their data are videos located in the following directories:
    * `/home/example/videos/Anno#1.mp4`
    * `/home/example/videos/Anno#1.mp4` 
4.  Also suppose that we have an appropriate meta-file (as described above).
5.  Firstly we need to check if there are the same people in both of annotations.
For that we will create a new task for these two annotations where will be
manually cross-checked tracks and marked as "the same". The command:
```bash
python3 main.py \
  --output_dir /home/example/output_dataset \
  --parse_only Anno \  # not required argument in our case where only one dataset is available
  --reid_model /path/to/person-reidentification-retail-0300.xml \  # here can be another model
  --split_dirs_by_source \  # this argument is strongly recommended
  --find_similar_tracks \
  --samples_proc_num 16 \
  --topk_closest_tracks 10 \
```
6. As result we get the next directories:
```
/home/example/output_dataset
├── Anno
│   ├── 000000_000
│   ├── 000000_001
│   ...
└── Anno_similar_tracks_v2
    └── scene_0
        ├── so#Anno#1.xml-id#0000077_000__0__s1#Anno#2.xml-id#0000019_001.jpg
        ...
```
7.  We use images from the directory `Anno_similar_tracks_v2` for the task
to match tracks. This task must be processed in [CVAT](https://github.com/opencv/cvat). Attributes in the task are the following:
```json
[
  {
    "name": "image",
    "id": 43692,
    "attributes": [
      {
        "id": 37827,
        "name": "merge",
        "type": "select",
        "mutable": true,
        "values": [
          "__undefined__",
          "yes",
          "no",
          "ignore"
        ]
      },
      {
        "id": 37826,
        "name": "quality_upper_track",
        "type": "select",
        "mutable": true,
        "values": [
          "__undefined__",
          "yes",
          "no"
        ]
      },
      {
        "id": 37825,
        "name": "quality_bottom_track",
        "type": "select",
        "mutable": true,
        "values": [
          "__undefined__",
          "yes",
          "no"
        ]
      }
    ]
  }
]
```
8. When annotation is finished we get the next *.xml file `Anno_tracks_comparison.xml`
(name can be different). So, we should manually add this annotation to our repository:
```
mcmt_extra
├── Anno
│   ├── meta.json
│   └── Anno.zip
│   │   ├── Anno#1.xml
│   │   └── Anno#2.xml
│   └── tracks_comparison.zip
│       └── Anno_tracks_comparison.xml
...
```
and update meta-file with this new annotation. For both of original annotations
in `meta.json` we add the next line:
```
"tracks_comparison": "tracks_comparison.zip/Anno_tracks_comparison.xml"
```

9. The next step is to patch our original *.xml annotations. Do it with the next command:
```bash
python3 main.py \
  --parse_only Anno \
  --patch_xml
``` 
10. As result our repository will be updated with two new directories:
`original_annotations` and `patched_annotations`. Patched annotations are our
targets. Its names are ended with `_patched` and looks like `Anno#1_patched.xml`.
We should manually replace our original annotations in `Anno.zip` by new patched
ones and the repository will be the next:
```
mcmt_extra
├── Anno
│   ├── meta.json
│   └── Anno.zip
│   │   ├── Anno#1_patched.xml
│   │   └── Anno#2_patched.xml
│   └── tracks_comparison.zip
│       └── Anno_tracks_comparison.xml
...
```
Names of annotations must replaced in meta-file too. 
Directories `original_annotations` and `patched_annotations` now are useless
and can be removed.
11. We done all necessary steps to make IDs of tracks unique for different people
and the same for merged ones. The last step creates output reid dataset. If we use
the same output_directory we should delete created early directory with cropped bounding boxes:
`/home/example/output_dataset/Anno` or set another output directory. The command
that creates target dataset in format of `Market1501` one:
```bash
python3 main.py \
  --output_dir /home/example/output_dataset \
  --parse_only Anno \  # again, not required in our case
  --market1501_format \
``` 
12. Our final result:
```
/home/example/output_dataset
├── Anno
│   ├── 000000
│   ├── 000001
│   ...
└── market1501_format
│   ├── bounding_box_test
│   ├── bounding_box_train
│   ├── bounding_box_val
│   ├── query
│   ├── query_val
├── test_gallery.txt
├── test_query.txt
├── train.txt
├── val_gallery.txt
└── val_query.txt
```
> *Note: .txt files consist of filtered images from the directory `Anno`, structure of `market1501_format` directory includes subdirectories with `val` that not used in original `Market 1501` dataset*
