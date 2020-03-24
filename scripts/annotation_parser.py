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

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import cv2
import logging as log
from lxml import etree
import os
from os import listdir
import os.path as osp
import re
from shutil import copyfile
from tqdm import tqdm

from .utils.filter_dark_light import filter_dark_light_image
from .utils.misc import *


class AnnotationParser(object):
    """Parsing CVAT annotation from *.xml files
    """
    def __init__(self, root_dir, annotations, data_files, data_types, scenes, cam_ids,
                 frame_rates, filters, targets, tracks_comparison, output_dir, target_samples_num, query_part=0.2,
                 min_track_len=2, expand_bboxes=(0.0, 0.0), id_len=5, sc_len=2, fr_len=6, vs_len=3, split_dirs_by_source=False):
        self.root_dir = root_dir
        self.annotations = annotations  # List of *xml files to parse
        self.data_files = data_files  # List of paths to video files or directories with images
        self.data_types = data_types  # List of data types: video, images
        self.scenes = scenes  # List of number of scenes in annotations
        self.cam_ids = cam_ids  # List of camera IDs
        self.frame_rates = frame_rates  # List of frame rates according to CVAT tasks
        self.filters = filters  # List of filters to apply to annotation
        self.targets = targets  # List of target subdataset: train, test, val
        self.tracks_comparison = tracks_comparison
        self.output_dir = output_dir  # List of output directories to save bounding boxes
        self.dataset_name = self.output_dir.split('/')[-1]  # Name of current dataset, equal to name of directory
        self.split_dirs_by_source = split_dirs_by_source  # Split cropped bounding boxes according to its video id

        self.target_samples_num = target_samples_num
        self.query_part = query_part
        self.min_track_len = min_track_len
        self.expand_bboxes = expand_bboxes
        self.id_len = id_len  # Length of ID in image name
        self.sc_len = sc_len  # Length of scene number in image name
        self.fr_len = fr_len  # Length of frame number in image name
        self.vs_len = vs_len  # Length of video source number in image name

        self.tracks = []
        self.processed_video = None

        self.files_created = False
        self.target_to_s_map = {'train': 0, 'test': 1}
        self.s_to_target_map = {0: 'train', 1: 'test'}

        self.output = {  # Final result with image names that will be saved
            'train': [[]],
            'test': [[], []],
            'val': [[], []]
        }
        self.file_map = {  # Map of saved files
            'train': [osp.join(self.root_dir, 'train.txt')],
            'test': [osp.join(self.root_dir, 'test_query.txt'), osp.join(self.root_dir, 'test_gallery.txt')],
            'val': [osp.join(self.root_dir, 'val_query.txt'), osp.join(self.root_dir, 'val_gallery.txt')],
        }
        self.market1501_dir_map = {
            'train': ['bounding_box_train'],
            'test': ['query', 'bounding_box_test'],
            'val': ['query_val', 'bounding_box_val']
        }
        self.statistics = {
            'train': [[0, 0]],
            'test': [[0, 0], [0, 0]],
            'val': [[0, 0], [0, 0]],
        }
        self.view_map = {None: '00', 'front': '01', 'back': '02', 'right': '03', 'left': '04'}
        self.target_id_constant = 10000

    def parse_and_extract_bounding_boxes(self, only_parse_xml=False):
        scene_indexes = []
        # Get indexes of annotations which are in the same scene
        for scene in set(self.scenes):
            scene_indexes.append([i for i, sc in enumerate(self.scenes) if sc == scene])

        for indexes in scene_indexes:
            self.tracks = []
            # Parse annotations inside the scene
            for i in indexes:
                if self.data_types[i] == 'video':
                    self.tracks.append(self.parse_video(i))
                elif self.data_types[i] == 'video_from_images':
                    self.tracks.append(self.parse_video_from_images(i))
                else:
                    self.tracks.append(self.parse_images(i))
            if not only_parse_xml:
                self.extract_bounding_boxes(indexes)

    def filter_samples(self, save_bboxes=False, show_statistics=True):
        def recursive_sum_list(input_list):
            res = 0
            if isinstance(input_list, list) and len(input_list) > 0 and isinstance(input_list[0], list):
                for l in input_list:
                    res += recursive_sum_list(l)
            else:
                res = len(input_list)
            return res

        skip = 0
        id_dirs = sorted([dir for dir in listdir(self.output_dir) if osp.isdir(osp.join(self.output_dir, dir))])
        for id_dir in tqdm(id_dirs, '{}: Filtering samples...'.format(self.dataset_name)):
            files = sorted([osp.join(id_dir, f) for f in listdir(osp.join(self.output_dir, id_dir))
                            if osp.isfile(osp.join(self.output_dir, id_dir, f))])
            if len(files) < self.min_track_len:
                skip += 1
                continue
            if len(files) < self.target_samples_num:
                self.put_in_output(files)
                continue
            split = self.split_by_cams(files)
            cams = sum([1 for x in split if recursive_sum_list(x) > 0])
            videos = [sum([1 for v in x if recursive_sum_list(v) > 0]) for x in split]
            views = [[sum([1 for y in v if recursive_sum_list(y) > 0]) for v in x] for x in split]
            for i, s in enumerate(split):
                for j, v in enumerate(s):
                    for k, _ in enumerate(v):
                        samples_num = self.target_samples_num // cams // videos[i] // views[i][j] \
                            if len(split[i][j][k]) > 0 else 0
                        step = len(split[i][j][k]) // samples_num + 1 if samples_num > 0 else 1
                        self.put_in_output([split[i][j][k][f] for f in range(0, len(split[i][j][k]), step)])
        if show_statistics:
            log.info('\'{}\' statistics:\n'
                     '  identities: {}\n'
                     '  samples number:\n'
                     '    train: {}\n'
                     '    test:\n'
                     '      query: {}\n'
                     '      gallery: {}\n'
                     '    val:\n'
                     '      query: {}\n'
                     '      gallery: {}'
                     .format(self.dataset_name, len(id_dirs) - skip, len(self.output['train'][0]),
                             len(self.output['test'][0]), len(self.output['test'][1]),
                             len(self.output['val'][0]), len(self.output['val'][1])))
        if save_bboxes:
            self.save_filtered_bboxes()

    def put_in_output(self, file_list):
        # Save file name in format: dataset_name/ID/sample_name.jpg
        if len(file_list) == 0:
            return
        query_step = int(len(file_list) / (len(file_list) * self.query_part)) + 1
        query = [file_list[i] for i in range(0, len(file_list), query_step)]
        for f in file_list:
            target = self.s_to_target_map[int(f.split('s')[1].split('_')[0])]
            output_file_name = osp.join(self.dataset_name, f)
            if target == 'train':
                self.output[target][0] += [output_file_name]
            else:
                if f in query:
                    self.output[target][0] += [output_file_name]
                else:
                    self.output[target][1] += [output_file_name]

    def parse_video(self, anno_id):
        ids = {}
        anno = self.annotations[anno_id]
        tree = etree.parse(anno)
        root = tree.getroot()
        for track_xml_subtree in tqdm(root, desc='{}: Reading {}'.format(self.dataset_name, anno.name)):
            if track_xml_subtree.tag != 'track':
                continue
            if track_xml_subtree.attrib['label'] == 'ignore':
                continue
            track_id = int(track_xml_subtree.attrib['id'])
            track_len = len(track_xml_subtree.findall('box')) - 1
            for t, box_tree in enumerate(track_xml_subtree.findall('box')):
                if t == track_len:
                    break
                track = self.parse_box_tree(box_tree, self.filters[anno_id])
                track['frame'] = int(box_tree.get('frame'))
                if len(track['bbox']) and track_id >= 0:
                    track['id'] = track_id
                else:
                    track['id'] = None
                if track['id'] is not None and len(track['bbox']) > 0 and len(track['view']) > 0:
                    ids = self.update_ids_with_track(ids, track, self.cam_ids[anno_id])
        return ids

    def parse_images(self, anno_id):
        ids = {}
        anno = self.annotations[anno_id]
        tree = etree.parse(anno)
        root = tree.getroot()
        for track_xml_subtree in tqdm(root, desc='{}: Reading {}'.format(self.dataset_name, anno.name)):
            if track_xml_subtree.tag != 'image':
                continue
            image_name = track_xml_subtree.attrib['name'].split('/')[-1]
            for box_tree in track_xml_subtree.findall('box'):
                track = self.parse_box_tree(box_tree, self.filters[anno_id])
                track['frame'] = image_name
                if track['id'] is not None and len(track['bbox']) > 0 and len(track['view']) > 0:
                    ids = self.update_ids_with_track(ids, track, self.cam_ids[anno_id])
        return ids

    def parse_video_from_images(self, anno_id):
        ids = {}
        anno = self.annotations[anno_id]
        tree = etree.parse(anno)
        root = tree.getroot()
        for track_xml_subtree in tqdm(root, desc='{}: Reading {}'.format(self.dataset_name, anno.name)):
            if track_xml_subtree.tag != 'track':
                continue
            if track_xml_subtree.attrib['label'] == 'ignore':
                continue
            track_id = int(track_xml_subtree.attrib['id'])
            track_len = len(track_xml_subtree.findall('box')) - 1
            for t, box_tree in enumerate(track_xml_subtree.findall('box')):
                if t == track_len:
                    break
                track = self.parse_box_tree(box_tree, self.filters[anno_id])
                track['frame'] = int(box_tree.get('frame')) * self.frame_rates[anno_id]
                if len(track['bbox']) and track_id >= 0:
                    track['id'] = track_id
                if track['id'] is not None and len(track['bbox']) > 0 and len(track['view']) > 0:
                    ids = self.update_ids_with_track(ids, track, self.cam_ids[anno_id])
        return ids

    def parse_box_tree(self, box_tree, filters):
        track = {'id': None, 'bbox': [], 'frame': [], 'view': [], 'occlusion': [], 'pose': []}
        # Get `occlusion` attribute
        occlusion = [tag.text for tag in box_tree if tag.attrib['name'] == 'occlusion']
        if len(occlusion) != 1:
            occlusion = [None]
        if 'occlusion' in filters.keys() and occlusion[0] in filters['occlusion']:
            return track
        # Get `view` attribute
        view = [tag.text for tag in box_tree if tag.attrib['name'] in ['view', 'orientation']]
        if len(view) != 1:
            view = [None]
        if 'view' in filters.keys() and view[0] in filters['view']:
            return track
        # Get `pose` attribute
        pose = [tag.text for tag in box_tree if tag.attrib['name'] == 'pose']
        if len(pose) != 1:
            pose = [None]
        if 'pose' in filters.keys() and pose[0] in filters['pose']:
            return track
        # Get ID of object
        id = [tag.text for tag in box_tree if tag.attrib['name'] == 'id']
        self.check_track_id(track, id)
        # Get coordinates of bounding box
        x_left, x_right = int(float(box_tree.get('xtl'))), int(float(box_tree.get('xbr')))
        y_top, y_bottom = int(float(box_tree.get('ytl'))), int(float(box_tree.get('ybr')))
        if x_right <= x_left or y_bottom <= y_top:
            return track
        # Check square of bounding box
        s = (x_right - x_left) * (y_bottom - y_top)
        if 'min_square' in filters.keys() and s < filters['min_square']:
            return track
        # Create output dict
        track['bbox'].append([x_left, y_top, x_right, y_bottom])
        track['view'] = view
        track['pose'] = pose
        track['occlusion'] = occlusion
        assert len(track['bbox']) == len(track['view']) == len(track['pose']) == len(track['occlusion']) == 1
        return track

    @staticmethod
    def check_track_id(track, track_id):
        if len(track_id) != 1:
            return
        try:
            track['id'] = int(track_id[0])
            if int(track['id']) < 0:
                track['id'] = None
        except:
            track['id'] = None

    @staticmethod
    def update_ids_with_track(ids, track, cam_id):
        if track['frame'] in ids.keys():
            ids[track['frame']][0] += [track['id']]
            ids[track['frame']][1] += track['bbox']
            ids[track['frame']][2] += track['view']
            ids[track['frame']][3] += [cam_id]
        else:
            ids[track['frame']] = [[track['id']], track['bbox'], track['view'], [cam_id]]
        assert len(ids[track['frame']][0]) == len(ids[track['frame']][1]) == \
               len(ids[track['frame']][2]) == len(ids[track['frame']][3])
        return ids

    def split_by_cams(self, file_list):
        cams = {k: v for v, k in enumerate(set(self.cam_ids))}
        output = [[[[], [], [], [], []] for __ in self.data_files] for _ in set(self.cam_ids)]
        for f in file_list:
            cam_id = cams[get_cam_id(f)]
            video_id = get_source_id(f, self.vs_len)
            view = get_view(f)
            output[cam_id][video_id][view].append(f)
        return output

    @staticmethod
    def int_to_str_with_len(value, length=6):
        value = str(value)
        while len(value) < length:
            value = '0' + value
        return value

    def make_id(self, n1, n2, prefix=None, len1=2, len2=5, prefix_len=2):
        result_id = self.int_to_str_with_len(n1, len1) + self.int_to_str_with_len(n2, len2)
        if prefix is not None:
            result_id = self.int_to_str_with_len(prefix, prefix_len) + result_id
        return result_id

    def get_image(self, anno_id, frame_counter=0, image_name=''):
        if self.data_types[anno_id] in ['video', 'video_from_images']:
            if self.processed_video is None:
                self.processed_video = cv2.VideoCapture(self.data_files[anno_id])
            has_frame, frame = self.processed_video.read()
            if has_frame:
                return frame, frame_counter + 1
        else:
            frame = cv2.imread(osp.join(self.data_files[anno_id], image_name))
        return frame, frame_counter + 1

    def extract_bounding_boxes(self, scene_indexes):
        """
        Crop bounding boxes from frames and save it with the next pattern:
        0011111_c2s3_444555555_66.jpg
        where:
          0 - number of scene
          1 - track id
          2 - camera id
          3 - target (0 - train, 1 - test)
          4 - video source number
          5 - frame number
          6 - view id
        """
        file_name_pattern = '{}_c{}s{}_{}_{}.jpg'
        for i, index in enumerate(scene_indexes):
            self.processed_video = None  # Reset video capture
            scene_num = self.scenes[index]
            sid_s = self.int_to_str_with_len(index, self.vs_len)
            frame_counter = -1
            # Go through all frames or images to extract bounding boxes and save it
            for c, (frame_num, values) in enumerate(tqdm(sorted(self.tracks[i].items()),
                                                         '{}: scene#{}, extracting bounding boxes, source {}/{}...'.
                                                         format(self.dataset_name, scene_num, i + 1, len(scene_indexes)))):
                if len(values[0]) == 0:
                    continue
                # Apply frame filtering
                if 'frames' in self.filters[index] and c % self.filters[index]['frames'] != 0:
                    continue
                # Choose image name
                if isinstance(frame_num, str):
                    frame_num_s = frame_num.split('_')[1].split('.')[0]
                    input_image_name = frame_num
                else:
                    frame_num_s = self.int_to_str_with_len(frame_num, self.fr_len)
                    input_image_name = 'frame_{}.jpg'.format(frame_num_s)

                # Get frames according to chosen frame rate in an annotation
                if self.data_types[index] in ['video', 'video_from_images']:
                    while frame_counter != self.get_valid_frame_num(frame_num_s):
                        input_frame, frame_counter = self.get_image(index, frame_counter)
                else:
                    input_frame, frame_counter = self.get_image(index, frame_counter, input_image_name)

                for j, id in enumerate(values[0]):
                    try:
                        # Create image ID: "ssddddd", ss = scene_num, ddddd = id
                        image_id = self.make_id(scene_num, id, len1=self.sc_len, len2=self.id_len)
                        save_dir = osp.join(self.output_dir, image_id)
                        if self.split_dirs_by_source:
                            save_dir += '_{}'.format(sid_s)
                        if not osp.exists(save_dir):
                            os.makedirs(save_dir)

                        bbox = values[1][j]
                        view = values[2][j]
                        cam_id = values[3][j]
                        target_index = self.target_to_s_map[self.targets[index]]
                        source_id_with_frame_number = sid_s + frame_num_s

                        file_name = file_name_pattern.format(image_id, cam_id, target_index,
                                                             source_id_with_frame_number, self.view_map[view])
                        ch = bbox[1] + (bbox[3] - bbox[1]) // 2
                        cw = bbox[0] + (bbox[2] - bbox[0]) // 2
                        h = int((bbox[3] - bbox[1]) * self.expand_bboxes[1])
                        w = int((bbox[2] - bbox[0]) * self.expand_bboxes[0])
                        x0, x1 = max(0, cw - w // 2), min(cw + w // 2, input_frame.shape[1])
                        y0, y1 = max(0, ch - h // 2), min(ch + h // 2, input_frame.shape[0])
                        crop = input_frame[y0: y1, x0: x1, :]
                        dl = filter_dark_light_image(crop)
                        if dl == 0:
                            save_file_name = osp.join(save_dir, file_name)
                            cv2.imwrite(save_file_name, crop)
                    except:
                        continue

    def get_valid_frame_num(self, frame_num):
        return int(frame_num)

    def create_train_query_and_gallery_files(self, rewrite=False):
        mode = 'w' if (rewrite and not self.files_created) else 'a'
        for target, image_list in self.output.items():
            for i, image_subset in enumerate(image_list):
                with open(self.file_map[target][i], mode) as f:
                    for image in image_subset:
                        f.write(image + "\n")
        self.files_created = True

    def save_like_market1501(self, dataset_name='market1501_format'):
        used_annos = {}
        counter = 0
        output_dir = osp.join(self.root_dir, dataset_name)
        for target, txt_files in tqdm(self.file_map.items(), 'Saving images in Market1501 format...'):
            for i, txt_file in enumerate(txt_files):
                target_dir = osp.join(output_dir, self.market1501_dir_map[target][i])
                if not osp.exists(target_dir):
                    os.makedirs(target_dir)
                with open(txt_file, 'r') as f:
                    lines = f.readlines()
                for l in lines:
                    anno_name = l.split('/')[0]
                    if anno_name not in used_annos.keys():
                        used_annos[anno_name] = counter
                        counter += 1
                    prefix = self.int_to_str_with_len(used_annos[anno_name], self.sc_len)
                    target_name = prefix + l.split('/')[-1].replace('\n', '')
                    target_file = osp.join(target_dir, target_name)
                    if not osp.exists(target_file):
                        copyfile(osp.join(self.root_dir, l.replace('\n', '')), target_file)
                    else:
                        log.warning('File \'{}\' already exists! Skipped'.format(target_file))
        log.info('Dataset in Market1501 format saved to \'{}\''.format(output_dir))

    def show_statistics(self, extended=True, base_distribution=10):
        def append(dict_obj, key, value):
            if key in dict_obj:
                if isinstance(value, dict):
                    for k, v in value.items():
                        append(dict_obj[key], k, v)
                else:
                    dict_obj[key] += value
            else:
                dict_obj[key] = value

        for subset_type, file_name in self.file_map.items():
            for i, sub_type in enumerate(file_name):
                if subset_type == 'test':
                    subset_name = 'query' if i == 0 else 'gallery'
                else:
                    subset_name = 'train'
                if not osp.isfile(self.file_map[subset_type][i]):
                    log.warning('File \'{}\' not found and will be skipped'.format(self.file_map[subset_type][i]))
                    continue
                with open(self.file_map[subset_type][i], 'r') as f:
                    lines = f.readlines()
                self.statistics[subset_type][i] = [
                    len(set([l.replace(l.split('/')[-1], '') for l in lines])),
                    len(lines)
                ]
                if extended:
                    nums = {}
                    views = {}
                    extended = {'num': {}, 'view': {}}
                    for l in lines:
                        unique_id = l.split('/')[0] + l.split('/')[1]
                        view = get_view(l)
                        append(nums, unique_id, 1)
                        append(views, unique_id, {view: 1})
                    for tid, num in nums.items():
                        append(extended['num'], num, 1)
                    for tid, view in views.items():
                        append(extended['view'], len(view), 1)
                    view_line = ''
                    for views_num, id_nums in extended['view'].items():
                        view_line += '\n\tidentities with views = {}: {}'.format(views_num, id_nums)
                    num_line = ''
                    left_edge, right_edge = 0, base_distribution
                    sum = 0
                    step = 0
                    for samples_num, id_nums in extended['num'].items():
                        left_edge, right_edge = base_distribution * step, base_distribution * (step + 1)
                        if left_edge <= samples_num < right_edge:
                            sum += id_nums
                        else:
                            num_line += '\n\tidentities with images number in [{}, {}): {}'.\
                                format(left_edge, right_edge, sum)
                            step += 1
                            sum = id_nums
                    num_line += '\n\tidentities with images number in [{}, {}): {}'.\
                        format(left_edge, right_edge, sum)
                    if len(view_line) > 0 and len(num_line) > 0:
                        log.info('{} subset extended statistics:\nviews:{}\nsamples_number{}'.
                                 format(subset_name, view_line, num_line))

        log.info('Result statistics:\n'
                 '  TRAIN:\n'
                 '    identities: {}\n'
                 '    samples number: {}\n'
                 '  TEST:\n'
                 '    query:\n'
                 '      identities: {}\n'
                 '      samples number: {}\n'
                 '    gallery:\n'
                 '      identities: {}\n'
                 '      samples number: {}\n'
                 '  VAL:\n'
                 '    query:\n'
                 '      identities: {}\n'
                 '      samples number: {}\n'
                 '    gallery:\n'
                 '      identities: {}\n'
                 '      samples number: {}'
                 .format(self.statistics['train'][0][0], self.statistics['train'][0][1], self.statistics['test'][0][0],
                         self.statistics['test'][0][1], self.statistics['test'][1][0], self.statistics['test'][1][1],
                         self.statistics['val'][0][0], self.statistics['val'][0][1], self.statistics['val'][1][0],
                         self.statistics['val'][1][1]))

    def save_filtered_bboxes(self):
        output_dir = self.output_dir.replace(self.dataset_name, '{}_bboxes'.format(self.dataset_name))
        if not osp.exists(output_dir):
            os.makedirs(output_dir)
        for target, subset in self.output.items():
            for i in range(len(subset)):
                for f in self.output[target][i]:
                    name = f.split(self.dataset_name + '/')[1]
                    copyfile(osp.join(self.output_dir, name), osp.join(output_dir, f.split('/')[-1]))
        log.info('{}: Filtered bounding boxes saved to: \'{}\''.format(self.dataset_name, output_dir))

    def extract_frames_from_videos(self, every_frame=False):
        for i, video in enumerate(self.data_files):
            if self.data_types[i] != 'video':
                continue
            frame_rate = self.frame_rates[i] if not every_frame else 1
            log.info('Extracting frames from video \'{}\'...'.format(video))
            cap = cv2.VideoCapture(video)
            video_name = osp.basename(video).split('.')[0]
            output_dir = osp.join(self.root_dir, '{}_frames/{}'.format(self.dataset_name, video_name))
            if not osp.exists(output_dir):
                os.makedirs(output_dir)
            has_frame = True
            num = 0
            while has_frame:
                has_frame, frame = cap.read()
                if has_frame and num % frame_rate == 0:
                    save_file_name = osp.join(output_dir, 'frame_{}.jpg'.
                                              format(self.int_to_str_with_len(num, self.fr_len)))
                    cv2.imwrite(save_file_name, frame)
                num += 1

    def merge_tracks_and_patch_xml(self, filter_bad_quality=False, output_dir='', save_statistics=True):
        for scene_num, anno_compare in self.tracks_comparison.items():
            similarity_matrix, bad_quality = self.parse_track_comparison_annos(anno_compare, filter_bad_quality)
            for i, anno in enumerate(self.annotations):
                if scene_num != self.scenes[i]:
                    continue
                pattern = self.get_pattern(i)
                txt = anno.read()
                save_name = anno.name.split('/')
                anno_base_name = save_name[-1]
                save_name[-1] = save_name[-1].split('.xml')[0] + '_patched.xml'
                save_name = osp.join(output_dir, save_name[-1])
                if save_statistics:
                    stat_str = '{}:'.format(anno_base_name)
                    badq_str = '{} -> Removed tracks (marked as with bad quality):'.format(anno_base_name)
                else:
                    stat_str = None
                    badq_str = None

                txt, badq_str, changes_are_made = self.mark_missed_ids(anno_base_name, similarity_matrix, txt,
                                                                       badq_str, pattern, bad_quality, i)

                for target_id_, entry in enumerate(tqdm(similarity_matrix, '{}: replacing IDs...'.format(anno_base_name))):
                    target_id = target_id_ + self.target_id_constant
                    if stat_str is not None:
                        stat_str += '\nTracks merged to ID = {}: '.format(target_id)
                    for j in range(0, len(entry)):
                        if entry[j].split('#')[0].endswith(anno_base_name):
                            current_id = int(entry[j].split('#')[1])
                            if entry[j] in bad_quality:
                                txt = txt.replace(pattern.format(current_id), pattern.format('-1'))
                                if badq_str is not None:
                                    badq_str += '\n{}'.format(current_id)
                            else:
                                txt = txt.replace(pattern.format(current_id), pattern.format(target_id))
                                if stat_str is not None:
                                    stat_str += '{}, '.format(current_id)
                            changes_are_made = True
                if changes_are_made:
                    if not osp.exists(output_dir):
                        os.makedirs(output_dir)
                    with open(save_name, 'w') as patched_file:
                        patched_file.write(txt)
                    if stat_str is not None:
                        basedir = save_name.split('/')
                        stat_file = osp.join('/', *basedir[:-1], 'merged_tracks_info.txt')
                        badq_file = osp.join('/', *basedir[:-1], 'removed_tracks_info.txt')
                        mode = 'w' if i == 0 else 'a'
                        with open(stat_file, mode) as f:
                            f.write('\n----------------\n' + stat_str)
                        with open(badq_file, mode) as f:
                            f.write('\n----------------\n' + badq_str)
                    log.info('Patched *.xml file saved to {}'.format(save_name))

    def get_pattern(self, anno_id=None):
        return 'track id=\"{}\" label=\"person\"'

    def get_all_ids(self, txt, anno_id=None):
        return re.findall("track id=\"(.{1,5})\" label=\"person\"", txt)

    def mark_missed_ids(self, anno_name, similarity_matrix, txt, badq_str, pattern, bad_quality, anno_id=None):
        used_ids = []
        changes_are_made = False
        for entries in similarity_matrix:
            for entry in entries:
                if entry.split('#')[0].endswith(anno_name):
                    used_ids.append(int(entry.split('#')[1]))
        all_track_ids = set(self.get_all_ids(txt, anno_id))
        all_track_ids = [int(x) for x in all_track_ids]
        for track_id in tqdm(all_track_ids, 'Marking missed IDs...'):
            if track_id not in used_ids:
                txt = txt.replace(pattern.format(track_id), pattern.format('-1'))
                if badq_str is not None:
                    badq_str += '\n{}'.format(track_id)
                changes_are_made = True
        for entry in bad_quality:
            track_id = int(entry.split('#')[1])
            txt = txt.replace(pattern.format(track_id), pattern.format('-1'))
            if badq_str is not None:
                badq_str += '\n{}'.format(track_id)
            changes_are_made = True
        return txt, badq_str, changes_are_made

    def check_collisions(self, similarity_matrix):
        dropped_tracks = []
        ids = [[] for _ in similarity_matrix]
        for i, tracks in enumerate(similarity_matrix):
            for track in tracks:
                for k, entry in enumerate(similarity_matrix):
                    if k > i and track in entry:
                        ids[i].append(k)
        ids = [list(set(x)) for x in ids]
        for i, id_list in enumerate(ids):
            if len(id_list):
                dropped_tracks += similarity_matrix[i]
        updated_similarity_matrix = [similarity_matrix[i] for i in range(len(ids)) if not len(ids[i])]
        return updated_similarity_matrix, dropped_tracks

    def parse_track_comparison_annos(self, anno, filter_bad_quality=True, extend_single_tracks=True):
        similarity_matrix = []
        bad_quality = set()
        all_track_ids = set()
        tree = etree.parse(anno)
        root = tree.getroot()
        for track_xml_subtree in tqdm(root, desc='{}: Reading annotation'.format(self.dataset_name)):
            if track_xml_subtree.tag != 'image':
                continue
            image_name = track_xml_subtree.attrib['name'].split('/')[-1]
            target_anno = [image_name.split('#')[1].split('-id')[0]]
            target_anno += [image_name.split('#')[3].split('-id')[0]]
            id = self.get_id_from_comparison(image_name)
            for box_tree in track_xml_subtree.findall('box'):
                merge = [tag.text for tag in box_tree if tag.attrib['name'] == 'merge']
                quality_0 = [tag.text for tag in box_tree if tag.attrib['name'] == 'quality_upper_track']
                quality_1 = [tag.text for tag in box_tree if tag.attrib['name'] == 'quality_bottom_track']
                if len(merge) != 1 or len(quality_0) != 1 or len(quality_1) != 1:
                    continue
                found_bad_quality_track = False
                target_name_0 = target_anno[0] + '#' + id[0]
                target_name_1 = target_anno[1] + '#' + id[1]
                if extend_single_tracks:
                    all_track_ids.add(target_name_0)
                    all_track_ids.add(target_name_1)
                if filter_bad_quality:
                    if quality_0[0] == 'no':
                        bad_quality.add(target_name_0)
                        found_bad_quality_track = True
                    if quality_1[0] == 'no':
                        bad_quality.add(target_name_1)
                        found_bad_quality_track = True
                if merge[0] == 'yes' and not found_bad_quality_track:
                    need_new_entry = True
                    for entry in similarity_matrix:
                        if target_name_0 in entry and target_name_1 not in entry:
                            entry += [target_name_1]
                            need_new_entry = False
                            break
                        elif target_name_1 in entry and target_name_0 not in entry:
                            entry += [target_name_0]
                            need_new_entry = False
                            break
                        elif target_name_0 in entry and target_name_1 in entry:
                            need_new_entry = False
                            break
                    if need_new_entry:
                        similarity_matrix.append([target_name_0, target_name_1])
        similarity_matrix, dropped_tracks = self.check_collisions(similarity_matrix)
        bad_quality = set(list(bad_quality) + dropped_tracks)
        if extend_single_tracks:
            for track_id in tqdm(all_track_ids, 'Searching for single tracks...'):
                found = False
                for entry in similarity_matrix:
                    if track_id in entry or track_id in bad_quality:
                        found = True
                        break
                if not found:
                    similarity_matrix.append([track_id])
        return similarity_matrix, bad_quality

    def get_id_from_comparison(self, image_name):
        pid_positions = [(2, '__'), (4, '.')]
        ids = []
        for pid in pid_positions:
            id_0 = image_name.split('#')[pid[0]].split(pid[1])[0].split('_')[0]
            if len(id_0) == self.sc_len + self.id_len:
                ids.append(id_0[self.sc_len:])
            else:
                ids.append(id_0)
        return ids

