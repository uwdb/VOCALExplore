import math
from typing import List

from vfe import core
from vfe import features
from vfe.api.storagemanager import ClipInfoWithPath

def align_to_subclips_strict(subclips, clip_start, clip_end):
    # Special case: clip is after the last subclip. Align the start of the clip to the start of the last subclip
    # then return without iterating over all subclips.
    if clip_start > subclips[-1][1]:
        clip_start = subclips[-1][0]
        return clip_start, clip_end

    overlapping_subclips_deltas = []

    for i in range(len(subclips)):
        subclip = subclips[i]
        # If subclip is before the clip, skip ahead until we're overlapping or after the clip.
        if subclip[1] < clip_start:
            continue

        # We know the subclip is either overlapping or after the clip.
        # If the subclip is entirely after the clip, then no subclip overlaps.
        if subclip[0] > clip_end:
            if not overlapping_subclips_deltas:
                # Get the distance from the clip to the one after as well as the one before.
                # If i == 0, then this is the first feature and there is none before.
                distance_to_start_of_previous = clip_start - subclips[i-1][0] if i > 0 else float('inf')
                overlapping_subclips_deltas.append((distance_to_start_of_previous, subclips[i-1][0], clip_end))

                distance_to_end_of_next = subclip[1] - clip_end
                overlapping_subclips_deltas.append((distance_to_end_of_next, clip_start, subclip[1]))
            break

        # If the clip is not entirely before or after the subclip, then it overlaps.
        suggested_start = min(clip_start, subclip[0])
        suggested_end = max(clip_end, subclip[1])
        expanded_delta = (suggested_end - suggested_start) - (clip_end - clip_start)
        overlapping_subclips_deltas.append((expanded_delta, suggested_start, suggested_end))

    ordered_deltas = sorted(overlapping_subclips_deltas, key=lambda x: x[0])
    clip_start, clip_end = ordered_deltas[0][1], ordered_deltas[0][2]
    return clip_start, clip_end

def align_to_subclips_expand(subclips, clip_start, clip_end):
    og_start, og_end = clip_start, clip_end
    clip_duration = clip_end - clip_start
    updated_start = False
    updated_end = False
    for i in range(len(subclips)):
        # Update start_time to be the start time of the closest feature that starts before the clip.
        if not updated_start and subclips[i][0] > og_start and not math.isclose(subclips[i][0], og_start, abs_tol=1e-6):
            updated_start = True
            clip_start = subclips[i-1][0]

            if clip_duration <= (subclips[i-1][1] - subclips[i-1][0]):
                # If the clip is shorter than the feature, expand the end of the clip backwards
                # in time. Otherwise the clip will become significantly longer when we also expand
                # the end forwards in time to align with a clip.
                clip_end = subclips[i-1][1]
                updated_end = True
        # Update end_time to be the end time of the closest feature ending after the clip.
        if not updated_end and subclips[i][1] >= og_end:
            updated_end = True
            clip_end = subclips[i][1]
        if updated_start and updated_end:
            break

    # Don't worry about the case where updated_end is not True. if this happens, then there is no
    # feature that covers the end of the clip. In this case, we can't grow the clip to get an extra
    # feature covered. Keep the tail end of the clip for context, even though the user's label on this
    # section won't be used for model training.

    # If not updated_start, clip.start_time is after the last feature. Move the clip
    # back to the start time of the latest feature.
    if not updated_start:
        clip_start = subclips[-1][0]

    return clip_start, clip_end


def align_to_feature(feature_names: List[str], clip: ClipInfoWithPath):
    # When there are multiple feature names, the first one is used to align the rest.
    # So use that one to get the stride dict.
    stride_dict = features.utils.get_extractor(feature_names[0], return_class=True).dali_kwargs(feature_names[0])
    sequence_length = stride_dict['sequence_length']
    stride = stride_dict['stride']
    step = stride_dict['step']
    # feature_clips are sorted in order of ascending start time.
    feature_clips = core.video.get_clips(clip.vpath, sequence_length, stride, step)
    if not feature_clips:
        # If the video is too short to get a complete feature, feature_clips will be empty.
        # Don't worry about aligning it in this case.
        return clip

    clip_start, clip_end = align_to_subclips_strict(feature_clips, clip.start_time, clip.end_time)

    # Push out clip_start, clip_end by a fraction of a second to ensure the clip completely covers a feature
    # despite rounding errors.
    return ClipInfoWithPath(clip.vid, clip.vstart, max(0, clip_start - 0.01), clip_end + 0.01, clip.vpath)

if __name__ == '__main__':
    subclips = [
        [0.1, 1.5],
        [0.75, 2.25],
        [1.5, 3],
        [2.25, 3.75],
        [3, 7],
    ]

    def _check_align(start, end, expected_start, expected_end):
        returned_start, returned_end = align_to_subclips_strict(subclips, start, end)
        assert returned_start == expected_start, f'Expected start {expected_start}, got {returned_start}'
        assert returned_end == expected_end, f'Expected end {expected_end}, got {returned_end}'

    # Test cases where clip completely contains a feature.
    # First feature.
    _check_align(0.1, 1.5, 0.1, 1.5)
    _check_align(0.1, 1.6, 0.1, 1.6)
    _check_align(0, 1.5, 0, 1.5)
    _check_align(0, 1.7, 0, 1.7)

    # Last feature.
    _check_align(3, 7, 3, 7)
    _check_align(2.9, 7, 2.9, 7)

    # Middle feature.
    _check_align(1.5, 3, 1.5, 3)
    _check_align(1.5, 3.1, 1.5, 3.1)
    _check_align(1.4, 3, 1.4, 3)
    _check_align(1.4, 3.8, 1.4, 3.8)

    # Test cases where clip does not overlap with any feature.
    # First feature.
    _check_align(-2, -1, -2, 1.5)

    # Last feature.
    _check_align(7.1, 7.2, 3, 7.2)

    # Test cases where clip partially overlaps with a feature.
    # First feature.
    _check_align(0.5, 1.6, 0.1, 1.6)
    _check_align(0, 1.4, 0, 1.5)
    _check_align(0.1, 0.2, 0.1, 1.5)
    _check_align(1, 1.5, 0.1, 1.5)

    # Last feature.
    _check_align(3.1, 7.1, 3, 7.1)
    _check_align(2.7, 5, 2.25, 5)
    _check_align(5, 8, 3, 8)

    # Middle feature.
    _check_align(1, 1.5, 0.1, 1.5)
    _check_align(1.6, 2.3, 1.5, 3)
