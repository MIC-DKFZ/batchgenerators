# Copyright 2021 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
# and Applied Computer Vision Lab, Helmholtz Imaging Platform
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

import os
import pickle
import json
from typing import List, Union, Optional


def subdirs(folder: str, join: bool = True, prefix: Optional[str] = None, suffix: Optional[str] = None, sort: bool = True) -> List[str]:
    """
    Returns a list of subdirectories in a given folder, optionally filtering by prefix and suffix,
    and optionally sorting the results. Uses os.scandir for efficient directory traversal.

    Parameters:
    - folder: Path to the folder to list subdirectories from.
    - join: Whether to return full paths to subdirectories (if True) or just directory names (if False).
    - prefix: Only include subdirectories that start with this prefix (if provided).
    - suffix: Only include subdirectories that end with this suffix (if provided).
    - sort: Whether to sort the list of subdirectories alphabetically.

    Returns:
    - List of subdirectory paths (or names) meeting the specified criteria.
    """
    subdirectories = []
    with os.scandir(folder) as entries:
        for entry in entries:
            if entry.is_dir() and \
               (prefix is None or entry.name.startswith(prefix)) and \
               (suffix is None or entry.name.endswith(suffix)):
                dir_path = entry.path if join else entry.name
                subdirectories.append(dir_path)

    if sort:
        subdirectories.sort()

    return subdirectories


def subfiles(folder: str, join: bool = True, prefix: Optional[str] = None, suffix: Optional[str] = None, sort: bool = True) -> List[str]:
    """
    Returns a list of files in a given folder, optionally filtering by prefix and suffix,
    and optionally sorting the results. Uses os.scandir for efficient directory traversal,
    making it suitable for network drives.

    Parameters:
    - folder: Path to the folder to list files from.
    - join: Whether to return full file paths (if True) or just file names (if False).
    - prefix: Only include files that start with this prefix (if provided).
    - suffix: Only include files that end with this suffix (if provided).
    - sort: Whether to sort the list of files alphabetically.

    Returns:
    - List of file paths (or names) meeting the specified criteria.
    """
    files = []
    with os.scandir(folder) as entries:
        for entry in entries:
            if entry.is_file() and \
               (prefix is None or entry.name.startswith(prefix)) and \
               (suffix is None or entry.name.endswith(suffix)):
                file_path = entry.path if join else entry.name
                files.append(file_path)

    if sort:
        files.sort()

    return files


def nifti_files(folder: str, join: bool = True, sort: bool = True) -> List[str]:
    return subfiles(folder, join=join, sort=sort, suffix='.nii.gz')


def maybe_mkdir_p(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)


def load_pickle(file: str, mode: str = 'rb'):
    with open(file, mode) as f:
        a = pickle.load(f)
    return a


def write_pickle(obj, file: str, mode: str = 'wb') -> None:
    with open(file, mode) as f:
        pickle.dump(obj, f)


def load_json(file: str):
    with open(file, 'r') as f:
        a = json.load(f)
    return a


def save_json(obj, file: str, indent: int = 4, sort_keys: bool = True) -> None:
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)


def pardir(path: str):
    return os.path.join(path, os.pardir)


def split_path(path: str) -> List[str]:
    """
    splits at each separator. This is different from os.path.split which only splits at last separator
    """
    return path.split(os.sep)


# I'm tired of typing these out
join = os.path.join
isdir = os.path.isdir
isfile = os.path.isfile
listdir = os.listdir
makedirs = maybe_mkdir_p
os_split_path = os.path.split

# I am tired of confusing those
subfolders = subdirs
save_pickle = write_pickle
write_json = save_json
