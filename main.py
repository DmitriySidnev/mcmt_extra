"""
 Copyright (c) 2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import argparse
import json
import logging as log
import os
import os.path as osp
from os import listdir
import sys
import zipfile

from scripts.annotation_parser import AnnotationParser
from scripts.similarity_finder import SimilarityFinder
try:
    from scripts.misc import *
    MISC_ON = True
except:
    MISC_ON = False

log.basicConfig(stream=sys.stdout, format='%(levelname)s: %(asctime)s-> %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S', level=log.DEBUG)

IGNORED_DIRS = ['scripts', ]
SCRIPTS_DIR = 'scripts'
EXTRACT_SUBDIR = 'original_annotations'
PATCHED_SUBDIR = 'patched_annotations'
META_FILE = 'meta.json'
# Constants describe names of files
ID_LEN = 5
SC_LEN = 2
FR_LEN = 6
VS_LEN = 3


def save_meta_file(meta_file_name):
    if MISC_ON:  # Save meta-file if it is possible
        meta_file = open(meta_file_name, 'w')
        json.dump(meta, meta_file)
        meta_file.close()


def directory_is_valid(root_dir, dir_name):
    return osp.isdir(osp.join(root_dir, dir_name)) \
           and dir_name not in IGNORED_DIRS and not dir_name.startswith('.')


def file_is_valid(root_dir, file_name):
    return osp.isfile(osp.join(root_dir, file_name))


def main():
    """Process *.xml annotations and extract bounding boxes with its IDs"""
    parser = argparse.ArgumentParser(description='Annotation parser')
    parser.add_argument('--output_dir', type=str, default='', required=True,
                        help='Path to output directory')
    parser.add_argument('--parse_only', type=str, nargs='+', default='',
                        help='Parse only chosen datasets')
    parser.add_argument('--target_samples_num', type=int, default=64,
                        help='Target number of samples for identity')
    parser.add_argument('--min_track_len', type=int, default=2,
                        help='Minimum number of samples for identity')
    parser.add_argument('--query_part', type=float, default=0.2,
                        help='Percentage of query part in test subset')
    parser.add_argument('--save_filtered_bboxes', default=False, action='store_true',
                        help='Save filtered bounding boxes')
    parser.add_argument('--extract_frames', default=False, action='store_true',
                        help='Only Extract frames from videos and finish')
    parser.add_argument('--show_statistics_only', default=False, action='store_true',
                        help='Collect statistics from txt files and finish')
    parser.add_argument('--skip_extracting_bboxes', default=False, action='store_true',
                        help='Do not extract bounding boxes from videos or images')
    parser.add_argument('--split_dirs_by_source', default=False, action='store_true',
                        help='Split cropped bounding boxes according to its video id')
    parser.add_argument('--expand_bboxes', type=float, nargs='+', default=(1.0, 1.0),
                        help='Expand width and height of bounding boxes')
    parser.add_argument('--rewrite_output_txt_files', default=False, action='store_true',
                        help='Rewrite already saved txt files with train, query and gallery split')
    parser.add_argument('--market1501_format', default=False, action='store_true',
                        help='Save dataset in like Market1501 format')
    # Patch annotations with merged tracks mode
    parser.add_argument('--patch_xml', default=False, action='store_true',
                        help='Patch *.xml annotations with merged tracks')
    # Similar-tracks-searching mode
    parser.add_argument('--find_similar_tracks', default=False, action='store_true',
                        help='Find similar tracks in annotations using ReId-model')
    parser.add_argument('--samples_proc_num', type=int, default=32,
                        help='Max number of samples per identity to extract embeddings')
    parser.add_argument('--topk_closest_tracks', type=int, default=5,
                        help='Max number of topK closest tracks to find')
    parser.add_argument('--samples_vis_num', type=int, default=16,
                        help='Max number of samples per identity to save in images')
    parser.add_argument('--sample_size', type=int, nargs='+', default=(128, 256),
                        help='Size of sample\'s image to save')
    parser.add_argument('--reid_model', type=str, default='',
                        help='Path to ReId-model (to *.xml file)')
    parser.add_argument('-l', '--cpu_extension', type=str, default=None,
                        help='MKLDNN (CPU)-targeted custom layers.Absolute \
                                  path to a shared library with the kernels impl.')
    args = parser.parse_args()

    args_string = ''
    for arg in vars(args):
        args_string += '\n\t{}: {}'.format(arg, getattr(args, arg))
    log.info('Input parameters: {}'.format(args_string))

    if args.patch_xml and args.find_similar_tracks:
        log.error('Switched on both of \'patch_xml\' and \'find_similar_tracks\' modes. '
                  'Please chose only one of them or switch off both.')
        return 1

    current_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = current_dir.split(SCRIPTS_DIR)[0]
    root_dir = ''

    dirs_to_parse = args.parse_only if len(args.parse_only) else listdir(parent_dir)
    anno_dirs = [osp.join(parent_dir, dir) for dir in dirs_to_parse if directory_is_valid(parent_dir, dir)]
    anno_string = ''
    for dir in anno_dirs:
        anno_string += '\n\t{}'.format(dir)
    log.info('Will be processed the next datasets: {}'.format(anno_string))

    annotation_parser = None
    if args.find_similar_tracks:
        log.info('Run a \'Similar-tracks-searching\' mode')
        if len(args.reid_model) == 0:
            log.error('\'--reid_model\' is necessary but empty. Please set up a ReId-model')
            return 1
        similarity_finder = SimilarityFinder(args.output_dir, args.reid_model, args.cpu_extension,
                                             args.samples_proc_num, args.topk_closest_tracks,
                                             args.sample_size, args.samples_vis_num, SC_LEN, VS_LEN)
    elif args.patch_xml:
        log.info('Run a \'Patch xml\' mode')

    for dir in anno_dirs:
        log.info('Processing of \'{}\' started'.format(dir))
        meta_file_name = osp.join(dir, META_FILE)

        #save_meta_file(meta_file_name)

        if not file_is_valid('', meta_file_name):
            log.warning('File \'{}\' not found. Annotation \'{}\' will be skipped'. format(meta_file_name, dir))
            continue
        with open(osp.join(dir, META_FILE), 'r') as f:
            meta_data = json.load(f)

        annotations = []
        data = []
        frame_rates = []
        filters = []
        data_types = []
        scene_numbers = []
        cam_ids = []
        targets = []
        tracks_comparison = {}

        if not osp.exists(args.output_dir):
            os.makedirs(args.output_dir)

        for entry in meta_data:
            if 'ignore' in entry.keys() and entry['ignore']:
                continue
            if entry['zip_archive'] is not None:
                zip_archive = osp.join(dir, entry['zip_archive'])
                if not file_is_valid('', zip_archive):
                    log.warning('File \'{}\' not found. Annotation \'{}\' will be skipped'.
                                format(zip_archive, entry['name']))
                    continue
            else:
                xml_file = osp.join(dir, entry['name'])
                if len(entry['name']) and not file_is_valid('', xml_file):
                    log.warning('File \'{}\' not found. Annotation \'{}\' will be skipped'.
                                format(xml_file, entry['name']))
                    continue
            if not entry['data_type'] in ['video', 'images', 'video_from_images']:
                log.warning('Unexpected data type \'{}\'. Annotation \'{}\' will be skipped'.
                            format(entry['data_type'], entry['name']))
                continue
            location = osp.join(root_dir, entry['location'])
            if not (directory_is_valid('', location) or file_is_valid('', location)):
                log.warning('File \'{}\' not found. Annotation \'{}\' will be skipped'.
                            format(location, entry['name']))
                continue

            if args.patch_xml and 'tracks_comparison' not in entry.keys():
                continue

            if entry['zip_archive'] is not None:
                archive = zipfile.ZipFile(zip_archive, 'r')
                annotations.append(archive.open(entry['name']))
            else:
                archive = None
                if len(entry['name']):
                    annotations.append(open(xml_file, 'r'))
                else:
                    fake_anno = FakeAnnotation
                    fake_anno.name = 'Chandler'
                    annotations.append(fake_anno)
            data.append(location)
            frame_rates.append(entry['frame_rate'])
            data_types.append(entry['data_type'])
            scene_numbers.append(entry['scene'])
            cam_ids.append(entry['cam_id'])
            filters.append(entry['filter'])
            targets.append(entry['target'])
            if args.patch_xml and 'tracks_comparison' in entry.keys():
                zip_arch_with_tracks = None
                names = entry['tracks_comparison'].split('/')
                for i, name in enumerate(names):
                    if name.endswith('.zip'):
                        zip_arch_with_tracks = osp.join(*names[:i + 1])
                        if entry['tracks_comparison'].startswith('/'):
                            zip_arch_with_tracks = '/' + zip_arch_with_tracks
                        else:
                            zip_arch_with_tracks = osp.join(dir, zip_arch_with_tracks)
                        zip_arch_with_tracks = zipfile.ZipFile(zip_arch_with_tracks, 'r')
                        tracks_comparison[entry['scene']] = zip_arch_with_tracks.open(names[-1])
                        break
                if zip_arch_with_tracks is None:
                    tracks_comparison[entry['scene']] = osp.join(dir, entry['tracks_comparison'])
                if archive is not None:
                    extract_path = osp.join(dir, EXTRACT_SUBDIR)
                    archive.extract(entry['name'], path=extract_path)
                    annotations[-1] = open(osp.join(extract_path, entry['name']), 'r')

        assert len(annotations) == len(data) == len(frame_rates) == len(data_types) == \
               len(scene_numbers) == len(cam_ids) == len(filters) == len(targets)

        if len(annotations) == 0:
            log.warning('Annotation \'{}\' is empty and will be skipped'.format(dir))
            continue
        else:
            anno_string = ''
            for anno in annotations:
                anno_string += '\n\t{}'.format(anno.name)
            log.info('List of processed annotations: {}'.format(anno_string))

        out_dir = osp.join(args.output_dir, osp.basename(dir))

        annotation_parser = AnnotationParser(args.output_dir, annotations, data, data_types, scene_numbers,
                                             cam_ids, frame_rates, filters, targets, tracks_comparison, out_dir,
                                             args.target_samples_num, args.query_part, args.min_track_len, args.expand_bboxes,
                                             ID_LEN, SC_LEN, FR_LEN, VS_LEN, args.split_dirs_by_source)
        if args.extract_frames:
            annotation_parser.extract_frames_from_videos()
            return 0
        if args.show_statistics_only:
            annotation_parser.show_statistics()
            return 0
        if args.patch_xml:
            annotation_parser.merge_tracks_and_patch_xml(output_dir=osp.join(dir, PATCHED_SUBDIR))
            return 0
        annotation_parser.parse_and_extract_bounding_boxes(only_parse_xml=args.skip_extracting_bboxes)
        if args.find_similar_tracks:
            similarity_finder.set_data_and_reset(annotations)
            similarity_finder.get_embeddings(annotation_parser.dataset_name)
            similarity_finder.process_embeddings()
            return 0
        annotation_parser.filter_samples(args.save_filtered_bboxes)
        annotation_parser.create_train_query_and_gallery_files(args.rewrite_output_txt_files)

    if annotation_parser is not None:
        if args.market1501_format:
            annotation_parser.save_like_market1501()
        annotation_parser.show_statistics()


if __name__ == '__main__':
    main()
