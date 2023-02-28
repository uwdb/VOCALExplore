# Based on https://github.com/mryoo/pooled_time_series
# Paper: https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Ryoo_Pooled_Motion_Features_2015_CVPR_paper.pdf

from collections import namedtuple
import cv2
import itertools
import logging
import math
import numpy as np
import time
from typing import List

from vfe.core import consts, videoframe
from vfe.datasets.frame import AbstractStride, CreateStride

Point = namedtuple('Point', ['x', 'y'])

class PoT:
    def __init__(self, frame_width=consts.RESIZE_WIDTH, frame_height=consts.RESIZE_HEIGHT, levels=4):
        self.frame_width=frame_width
        self.frame_height=frame_height
        self.levels = levels

    def extract_series(self, video_filename, optical=True, gradient=True):
        if optical:
            optical_series = self.get_optical_time_series(video_filename, 5, 5, 8)
        else:
            optical_series= []

        if gradient:
            gradient_series = self.get_gradient_time_series(video_filename, 5, 5, 8)
        else:
            gradient_series = []
        return optical_series, gradient_series

    def extract_features(self, series):
        tws = self.get_temporal_windows(self.levels)
        fv = []
        fv.extend(self.compute_features_from_series(series, tws, 1))
        fv.extend(self.compute_features_from_series(series, tws, 2))
        fv.extend(self.compute_features_from_series(series, tws, 5))
        return fv

    def get_optical_time_series(self, filename, w_d: int, h_d: int, o_d: int):
        hists = self.get_optical_histograms(filename, w_d, h_d, o_d)
        vectors = [PoT.histogram_to_vector(hist) for hist in hists]
        return vectors

    def get_gradient_time_series(self, filename, w_d: int, h_d: int, o_d: int):
        hists = self.get_gradient_histograms(filename, w_d, h_d, o_d)
        vectors = [PoT.histogram_to_vector(hist) for hist in hists]
        return vectors

    @staticmethod
    def histogram_to_vector(hist):
        return hist.ravel(order='C')

    def get_gradient_histograms(self, filename, w_d: int, h_d: int, o_d: int):
        histograms = []
        capture = cv2.VideoCapture(filename)
        if not capture.isOpened():
            logging.warn('Video file not opened')
            histograms.append(np.zeros((w_d, h_d, o_d), dtype=np.float64))
        else:
            original_frame = None
            resized = None
            resized_gray = None
            # initializing a list of histogram of gradients (i.e. a list of s*s*9
            # arrays)
            for i in itertools.count():
                # capturing the video images
                ret, original_frame = capture.read()
                if not ret:
                    if i == 0:
                        raise RuntimeError("Could not read the video file")
                    else:
                        break

                hist = np.zeros((w_d, h_d, o_d), dtype=np.float64)
                gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
                resized_gray = cv2.resize(gray, (self.frame_width, self.frame_height))

                # This is super slow.
                # DEBUG:root:perf: compute_gradients took 4.722782679978991 sec
                # DEBUG:root:perf: update_gradient_histogram took 0.5034153400047217 sec
                gradients = PoT.compute_gradients(resized_gray, o_d)
                PoT.update_gradient_histogram(hist, gradients)

                histograms.append(hist)
        capture.release()
        return histograms

    @staticmethod
    def compute_gradients(frame, dim: int):
        frame_array = frame.tobytes(order='C')

        gradients = []
        rows, cols = frame.shape
        for k in range(dim):
            angle: float = math.pi * float(k) / float(dim)
            dx: float = math.cos(angle) * 0.9999999
            dy: float = math.sin(angle) * 0.9999999

            grad = np.zeros((cols, rows), dtype=np.float64)
            for i in range(cols):
                for j in range(rows):
                    if i <= 1 or j <= 1 or i >= cols - 2 or j >= rows - 2:
                        grad[i][j] = 0
                    else:
                        f1: float = PoT.interpolate_pixel(frame_array, cols, float(i) + dx, float(j) + dy)
                        f2: float = PoT.interpolate_pixel(frame_array, cols, float(i) - dx, float(j) - dy)
                        diff: float = f1 - f2
                        if diff < 0:
                            diff = diff * -1
                        if diff >= 256:
                            diff = 255
                        grad[i][j] = diff
            gradients.append(grad)

        return gradients

    @staticmethod
    def interpolate_pixel(image, w: int, x: float, y: float):
        x1: float = float(int(x))
        x2: float = float(int(x) + 1)
        y1: float = float(int(y))
        y2: float = float(int(y) + 1)

        f11: float = float(image[int(y) * w + int(x)] & 0xFF)
        f21: float = float(image[int(y) * w + int(x) + 1] & 0xFF)
        f12: float = float(image[int(y + 1) * w + int(x)] & 0xFF)
        f22: float = float(image[int(y + 1) * w + int(x) + 1] & 0xFF)

        f: float = f11 * (x2 - x) * (y2 - y) + f21 * (x - x1) * (y2 - y) + f12 * (x2 - x) * (y - y1) + f22 * (x - x1) * (y - y1)

        return f

    @staticmethod
    def update_gradient_histogram(hist, gradients):
        d1, d2, d3 = hist.shape
        width, height = gradients[0].shape
        for i in range(width):
            s1_index: int = int(i * d1 / width)
            for j in range(height):
                s2_index: int = int(j * d2 / height)
                for k in range(d3):
                    val: float = gradients[k][i][j] / 100.0
                    hist[s1_index][s2_index][k] += val

    def get_optical_histograms(self, filename, w_d: int = 5, h_d: int = 5, o_d: int = 8, optimized=True, flow_args={}, stride: AbstractStride=None):
        return self.get_optical_histograms_from_info(videoframe.extract_raw_frames(filename), w_d=w_d, h_d=h_d, o_d=o_d, optimized=optimized)

    def get_optical_histograms_from_info(self, video_info: videoframe.VideoInfo, w_d: int = 5, h_d: int = 5, o_d: int = 8, optimized=True, flow_args={}, stride: AbstractStride=None):
        assert video_info.width == self.frame_width
        assert video_info.height == self.frame_height
        histograms = []
        if stride is None:
            stride = CreateStride(fstride=1)
        step = stride.step(video_info.fps, video_info.nframes)

        cv2_time = 0
        update_time = 0

        # variables for processing images
        original_frame = None
        frame_gray = None
        prev_frame_gray = None
        flow = None

        # computing a list of histogram of optical flows (i.e. a list of 5*5*8
        # arrays)
        for frame_index in range(0, video_info.max_frame, step):
            original_frame = video_info.frames[frame_index]
            frame_gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)

            hist = np.zeros((w_d, h_d, o_d), dtype=np.float64)
            histograms.append(hist)

            # from frame #2
            if frame_index > 0:
                # calculate optical flows
                start_cv2 = time.perf_counter()
                flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, frame_gray, None, 0.5, 1, 10, 2, 7, 1.5, 0) # 0.5, 1, 15, 2, 7, 1.5, 0
                # cv2_time += time.perf_counter() - start_cv2

                # update histogram of optical flows
                start_update = time.perf_counter()
                if optimized:
                    self.updateOpticalHistogramOptimized(histograms[-1], flow, **flow_args)
                else:
                    self.updateOpticalHistogram(histograms[-1], flow)
                # update_time += time.perf_counter() - start_update

            temp_frame = prev_frame_gray
            prev_frame_gray = frame_gray
            frame_gray = temp_frame

        # logging.debug(f'cv2 time: {cv2_time}, update time: {update_time}')
        return histograms

    def updateOpticalHistogramOptimized(self, hist, flow, step=4):
        d1, d2, d3 = hist.shape
        # Do we need to step if we've vectorized?
        sizes = np.sum(flow, axis=2)
        sizes = np.power(sizes, 2)
        degrees = np.arctan2(flow[..., 1], flow[..., 0])
        # Compute flow type.
        boundaries = np.arange(d3)
        boundaries = (boundaries + 1) * 2.0 * math.pi / d3 - math.pi
        default_type = 7
        types = np.ones_like(sizes) * default_type
        # We ignore cases where the size is small. Set these indices to a flag value that we can ignore later.
        types[np.where(sizes < 9)] = -1
        for i, boundary in enumerate(boundaries):
            types[np.where((degrees < boundary) & (types == default_type))] = i
        # For each window, update hist.
        x_lim = lambda x_type, dir: int(self.frame_width * (x_type + (0 if dir == 'min' else 1)) / d1)
        y_lim = lambda y_type, dir: int(self.frame_height * (y_type + (0 if dir == 'min' else 1)) / d2)
        min_xs = [x_lim(xt, 'min') for xt in range(d1)]
        max_xs = [x_lim(xt, 'max') for xt in range(d1)]
        min_ys = [y_lim(yt, 'min') for yt in range(d2)]
        max_ys = [y_lim(yt, 'max') for yt in range(d2)]
        for x_type in range(d1):
            for y_type in range(d2):
                types_window = types[min_ys[y_type]:max_ys[y_type], min_xs[x_type]:max_xs[x_type]].astype(np.int8)
                types_window = types_window[::step, ::step]
                hist[x_type][y_type] += np.bincount(types_window[types_window != -1].flatten(), minlength=d3)

    def updateOpticalHistogram(self, hist, flow):
        d1, d2, d3 = hist.shape
        step = 4 # 5

        for x in range(0, self.frame_width, step):
            x_type = int(x * d1 / self.frame_width)
            for y in range(0, self.frame_height, step):
                y_type = int(y * d2 / self.frame_height)
                fxy = Point(*flow[y, x])
                size = (fxy.x + fxy.y) * (fxy.x + fxy.y)
                if size < 9:
                    continue # 25
                else:
                    f_type = PoT.opticalFlowType(fxy, d3)

                hist[x_type][y_type][f_type] += 1

    @staticmethod
    def opticalFlowType(fxy: Point, dim: int):
        degree: float = math.atan2(fxy.y, fxy.x)
        type: int = 7
        for i in range(dim):
            boundary: float = (i + 1) * 2 * math.pi / dim - math.pi
            if degree < boundary:
                type = i
                break
        return type

    def get_temporal_windows(self, level):
        fws = []
        for l in range(level):
            cascade_steps = int(float(2) ** float(l))
            step_size = float(1) / float(cascade_steps)
            for k in range(cascade_steps):
                start = step_size * float(k) + 0.000001
                end = step_size * float(k + 1) + 0.000001
                wind = [start, end]
                fws.append(wind)
        return fws

    def compute_features_from_series(self, series, time_windows_list, feature_mode) -> List:
        start = 0
        end = len(series) - 1
        feature = []
        for j in range(len(time_windows_list)):
            duration = end - start
            for i in range(len(series[0])): # I think assumes that all series are the same length.
                if duration < 0: # Possible if series is empty.
                    if feature_mode == 2 or feature_mode == 4: # Feature modes with two output values.
                        feature.append(0.0)
                        feature.append(0.0)
                    else: # Feature mode with a single output value.
                        feature.append(0.0)
                    continue

                window_start = start + int(duration * time_windows_list[j][0] + 0.5)
                window_end = start + int(duration * time_windows_list[j][1] + 0.5) # window_end is inclusive.

                if feature_mode == 1: # sum pooling
                    sum = 0
                    for t in range(window_start, window_end+1):
                        if t < 0:
                            continue
                        sum += series[t][i]
                    feature.append(sum)
                elif feature_mode == 2: # gradient pooling1
                    positive_gradients = 0.0
                    negative_gradients = 0.0
                    for t in range(window_start, window_end+1):
                        look = 2
                        if t - look < 0:
                            continue
                        else:
                            dif = series[t][i] - series[t - look][i]
                            if dif > 0.01: # 0.01 for optical
                                positive_gradients += 1
                            elif dif < -0.01:
                                negative_gradients += 1
                    feature.append(positive_gradients)
                    feature.append(negative_gradients)
                elif feature_mode == 4: # gradient pooling2
                    positive_gradients = 0.0
                    negative_gradients = 0.0
                    for t in range(window_start, window_end+1):
                        look = 2
                        if t - look < 0:
                            continue
                        else:
                            dif = series[t][i] - series[t - look][i]
                            if dif > 0:
                                positive_gradients += dif
                            else:
                                negative_gradients += -dif
                    feature.append(positive_gradients)
                    feature.append(negative_gradients)
                elif feature_mode == 5: # max pooling
                    max = -1000000
                    for t in range(window_start, window_end+1):
                        if t < 0:
                            continue
                        if series[t][i] > max:
                            max = series[t][i]
                    feature.append(max)
        return feature

    def normalizeFeatureL1(self, sample):
        sum = 0
        for i in range(len(sample)):
            val = sample[i]
            if val < 0:
                val = -1 * val
            sum += val

        for i in range(len(sample)):
            if sum == 0:
                v = 0
            else:
                v = sample[i] / sum;# *100;
            sample[i] = v

    @staticmethod
    def chiSquareDistance(feature1, feature2):
        if len(feature1) != len(feature2):
            logging.warn("feature vector dimension mismatch.")
        score = 0
        for i in range(len(feature1)):
            h1 = feature1[i]
            h2 = feature2[i]
            if h1 < 0 or h2 < 0:
                logging.error("A negative feature value. The chi square kernel "
                    + "does not work with negative values. Please try shifting "
                    + "the vector to make all its elements positive.")
            if h1 == h2:
                continue
            else:
                score += (h1 - h2) * (h1 - h2) / (h1 + h2)

        return 0.5 * score

    @staticmethod
    def meanChiSquareDistances(samples, d):
        # samples: arraylist of FeatureVector type
        mean_dist = 0
        sum = 0.0
        count = 0
        for i in range(len(samples)):
            for j in range(i+1, len(samples)):
                count += 1
                sum += PoT.chiSquareDistance(samples[i].feature[i], samples[j].feature[d])
        mean_dist = sum / float(count)
        return mean_dist

    @staticmethod
    def kernelDistance(sample1, sample2, mean_dists):
        # both sample1 and sample2 are of type FeatureVector
        distance = 0.0
        for d in range(sample1.num_dim()):
            weight = 1.0
            if mean_dists[d] == 0:
                val = PoT.chiSquareDistance(sample1.feature[d], sample2.feature[d]) / 1000000.0
            else:
                val = PoT.chiSquareDistance(sample1.feature[d], sample2.feature[d]) / mean_dists[d] * weight
            distance = distance + val

        final_score = math.exp(-1 * distance / 10); # 10000 10
        return final_score
