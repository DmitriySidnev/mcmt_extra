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

import cv2

import logging as log
import numpy as np
import os
import os.path as osp
from collections import Counter
from os import listdir
from scipy.spatial.distance import cdist
from tqdm import tqdm

from .utils.misc import *
from .utils.openvino_wrapper import OpenVINOModel


NARROW_DELIM = 10
WIDE_DELIM = 30
BW = 3  # border width
BLACK = (0, 0, 0)
TEXT_HEADER_HEIGHT = 20
TEXT_SCALE = 0.3


class SimilarityFinder(object):
    def __init__(self, root_dir, reid_model, cpu_extension='', max_samples_num=32,
                 topk=5, sample_size=(128, 256), max_track_len=16, sc_len=2, vs_len=3):
        self.root_dir = root_dir
        self.reid_model = OpenVINOModel(reid_model, cpu_extension)
        self.dataset_name = ''
        self.max_samples_num = max_samples_num
        self.topk = topk
        self.sample_size = sample_size
        self.max_track_len = max_track_len
        self.sc_len = sc_len
        self.vs_len = vs_len
        self.data = []
        self.embeddings = {}  # scene_id: [(track_id, image, embedding), ...]

    def set_data_and_reset(self, new_data):
        self.data = new_data
        self.embeddings = {}

    def set_dataset_name(self, new_dataset_name):
        self.dataset_name = new_dataset_name

    def get_embeddings(self, dataset_name):
        self.set_dataset_name(dataset_name)
        work_dir = osp.join(self.root_dir, dataset_name)
        id_dirs = sorted([osp.join(work_dir, dir) for dir in listdir(work_dir) if osp.isdir(osp.join(work_dir, dir))])
        for id_dir in tqdm(id_dirs, '{}: Extracting embeddings'.format(dataset_name)):
            image_list = sorted([osp.join(id_dir, image) for image in listdir(id_dir)
                                 if osp.isfile(osp.join(id_dir, image))])
            step = len(image_list) // self.max_samples_num + 1
            batch = []
            scene_ids = []
            track_ids = []
            images = []
            for i in range(0, len(image_list), step):
                scene_ids.append(get_scene_id(image_list[i], self.sc_len))
                track_ids.append(image_list[i].split('/')[-2])
                images.append(image_list[i])
                batch.append(image_list[i])
            if len(batch) > 0:
                self.process_batch(batch, scene_ids, track_ids, images)

    def process_batch(self, batch, scene_ids, track_ids, images):
        embeddings = self.reid_model.forward(batch)
        assert len(scene_ids) == len(track_ids) == len(images) == len(embeddings)
        for i in range(len(track_ids)):
            if scene_ids[i] in self.embeddings:
                self.embeddings[scene_ids[i]].append((track_ids[i], images[i], embeddings[i]))
            else:
                self.embeddings[scene_ids[i]] = [(track_ids[i], images[i], embeddings[i])]

    def process_embeddings(self):
        for scene_id, values in self.embeddings.items():
            all_embeddings = []
            all_tracks = {}
            sample_description = []
            for track_id, image, embedding in values:
                all_embeddings.append(embedding)
                sample_description.append((track_id, image))
                if track_id in all_tracks:
                    all_tracks[track_id].append(image)
                else:
                    all_tracks[track_id] = [image]
            all_embeddings = np.squeeze(np.asarray(all_embeddings, dtype='float32'))
            log.info('{}: scene#{}, calculating distances...'.format(self.dataset_name, scene_id))
            dist_matrix = cdist(all_embeddings, all_embeddings, 'cosine')
            log.info('{}: scene#{}, searching for topK = {} closest tracks...'.format(self.dataset_name, scene_id, self.topk))
            topk_closest_tracks = self.find_topk_closest_tracks_v2(dist_matrix, sample_description)
            self.save_target_images(scene_id, topk_closest_tracks, all_tracks)

    def find_topk_closest_tracks(self, dist_matrix, id_desc):
        distance_indices = np.argsort(dist_matrix, axis=1)
        topk_tracks_per_sample = np.zeros((distance_indices.shape[0], self.topk), dtype='int32')
        # Find topK closest tracks per sample
        for i in range(distance_indices.shape[0]):
            found_ind = 0
            used_tracks = []
            for j in range(distance_indices.shape[1]):
                if id_desc[i][0] != id_desc[distance_indices[i][j]][0] and distance_indices[i][j] not in used_tracks:
                    topk_tracks_per_sample[i][found_ind] = distance_indices[i][j]
                    used_tracks.append(distance_indices[i][j])
                    found_ind += 1
                    if found_ind == self.topk:
                        break

        topk_closest_tracks = {}
        # Find topK the most frequently tracks per ID
        for i in range(topk_tracks_per_sample.shape[0]):
            tid = id_desc[i][0]
            for j in range(topk_tracks_per_sample.shape[1]):
                if tid in topk_closest_tracks:
                    topk_closest_tracks[tid].append(id_desc[topk_tracks_per_sample[i][j]][0])
                else:
                    topk_closest_tracks[tid] = [id_desc[topk_tracks_per_sample[i][j]][0]]
        for tid, closest_tracks in topk_closest_tracks.items():
            topk_closest_tracks[tid] = Counter(closest_tracks)
            topk = []
            topk_closest_tracks[tid] = sorted(topk_closest_tracks[tid].items(), key=lambda kv: kv[1], reverse=True)
            for i in range(min(self.topk, len(topk_closest_tracks[tid]))):
                topk.append(topk_closest_tracks[tid][i][0])
            topk_closest_tracks[tid] = topk

        return topk_closest_tracks

    def find_topk_closest_tracks_v2(self, dist_matrix, id_desc, min_num=2):
        distance_indices = np.argsort(dist_matrix, axis=1)
        topk_tracks_per_sample = []
        # Find topK closest tracks per sample
        for i in range(distance_indices.shape[0]):
            track_i_id = id_desc[i][0]
            found_ind = 0
            used_tracks = []
            pool = []
            for j in range(distance_indices.shape[1]):
                track_j_id = id_desc[distance_indices[i][j]][0]
                if track_i_id != track_j_id:
                    pool.append(distance_indices[i][j])
                    if track_j_id not in used_tracks:
                        used_tracks.append(track_j_id)
                        found_ind += 1
                        if found_ind == self.topk + 1:
                            break
            topk_tracks_per_sample.append(pool)

        topk_closest_tracks = {}
        # Find topK the most frequently IDs
        for i in range(len(topk_tracks_per_sample)):
            tid = id_desc[i][0]
            for j in range(len(topk_tracks_per_sample[i])):
                if tid in topk_closest_tracks:
                    topk_closest_tracks[tid].append(id_desc[topk_tracks_per_sample[i][j]][0])
                else:
                    topk_closest_tracks[tid] = [id_desc[topk_tracks_per_sample[i][j]][0]]
        for tid, closest_tracks in topk_closest_tracks.items():
            topk_closest_tracks[tid] = Counter(closest_tracks)
            topk = []
            topk_closest_tracks[tid] = sorted(topk_closest_tracks[tid].items(), key=lambda kv: kv[1], reverse=True)
            for i in range(min(self.topk, len(topk_closest_tracks[tid]))):
                if topk_closest_tracks[tid][i][1] >= min_num:
                    topk.append(topk_closest_tracks[tid][i][0])
            topk_closest_tracks[tid] = topk

        return topk_closest_tracks

    def save_target_images(self, scene_id, topk_closest_tracks, tracks, dir_suffix='_similar_tracks_v2'):
        target_dir = osp.join(self.root_dir, self.dataset_name + dir_suffix, 'scene_{}'.format(scene_id))
        if not osp.exists(target_dir):
            os.makedirs(target_dir)
        for tid_i, topk_tracks in tqdm(topk_closest_tracks.items(),
                                       '{}: scene#{}, saving images'.format(self.dataset_name, scene_id)):
            source_i_id = get_source_id(tracks[tid_i][0], self.vs_len)
            source_i = self.data[int(source_i_id)].name.split('/')[-1]
            img_top = self.make_image(tracks[tid_i], str(tid_i), NARROW_DELIM)
            for position, tid_j in enumerate(topk_tracks):
                source_j_id = get_source_id(tracks[tid_j][0], self.vs_len)
                source_j = self.data[int(source_j_id)].name.split('/')[-1]
                img_bottom = self.make_image(tracks[tid_j], str(tid_j), NARROW_DELIM)
                middle_delim = np.full((WIDE_DELIM, img_bottom.shape[1], img_bottom.shape[2]), 255, dtype='uint8')
                img_name = 's{}#{}-id#{}__{}__s{}#{}-id#{}.jpg'.\
                    format(source_i_id, source_i, tid_i, position, source_j_id, source_j, tid_j)
                target_image_path = osp.join(target_dir, img_name)
                if img_top.shape[1] != img_bottom.shape[1]:
                    width = min(img_top.shape[1], img_bottom.shape[1])
                    img_top_resized = img_top[:, :width, :]
                    img_bottom = img_bottom[:, :width, :]
                    middle_delim = middle_delim[:, :width, :]
                    target_image = np.vstack([img_top_resized, middle_delim, img_bottom])
                else:
                    target_image = np.vstack([img_top, middle_delim, img_bottom])
                cv2.imwrite(target_image_path, target_image)
        log.info('{}: Images saved to \'{}\''.format(self.dataset_name, target_dir))

    def make_image(self, track, track_id, delimiter_width, border_color=BLACK, samples_num=15):
        result_img = None
        width, height = self.sample_size
        step = len(track) // samples_num + 1
        for i in range(0, len(track), step):
            frame_num = track[i].split('/')[-1].split('_')[2]
            track_info = track_id + ': ' + frame_num
            img = cv2.imread(track[i])
            img = cv2.resize(img, self.sample_size)
            img = cv2.copyMakeBorder(img, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=border_color)
            # resize twice to ensure that the border width is consistent across images
            img = cv2.resize(img, self.sample_size)
            sample_img = np.full(img.shape, 255, dtype='uint8')
            sample_img = cv2.resize(sample_img, (width, height + TEXT_HEADER_HEIGHT))
            sample_img[TEXT_HEADER_HEIGHT:, :, :] = img
            label_size, _ = cv2.getTextSize(track_info, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, 1)
            text_point = (width // 2 - label_size[0] // 2, TEXT_HEADER_HEIGHT // 2 + label_size[1] // 4)
            cv2.putText(sample_img, track_info, text_point, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, BLACK, 1)
            delimiter_size = (sample_img.shape[0], delimiter_width, sample_img.shape[2])
            delimiter = np.full(delimiter_size, 255, dtype='uint8')
            sample_img = np.hstack([sample_img, delimiter])
            if result_img is None:
                result_img = sample_img
            else:
                result_img = np.hstack([result_img, sample_img])
        return result_img
