# Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from builtins import range
from warnings import warn

def data_channel_selection_generator(generator, selected_channels):
    '''
    yields selected channels from data
    '''
    warn("using deprecated generator data_channel_selection_generator", Warning)
    for data_dict in generator:
        assert "data" in list(
            data_dict.keys()), "your data generator needs to return a python dictionary with at least a 'data' key value pair"
        data = data_dict["data"]
        data_dict["data"] = data[:, selected_channels]
        yield data_dict


def seg_channel_selection_generator(generator, selected_channels, keep_discarded_seg=False):
    '''
    yields selected channels from seg
    '''
    warn("using deprecated generator seg_channel_selection_generator", Warning)
    for data_dict in generator:
        do_seg = False
        seg = None
        if "seg" in list(data_dict.keys()):
            seg = data_dict["seg"]
            do_seg = True
        if not do_seg:
            Warning(
                "You used center_crop_seg_generator but there is no 'seg' key in your data_dict, returning data_dict unmodified")
            yield data_dict
        data_dict["seg"] = seg[:, selected_channels]
        if keep_discarded_seg:
            discarded_seg_idx = [i for i in range(len(seg[0])) if i not in selected_channels]
            data_dict['discarded_seg'] = seg[:, discarded_seg_idx]
        yield data_dict
