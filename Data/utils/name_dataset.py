from __future__ import print_function
import torch
from os import listdir
from os.path import join, isfile
from Data.utils.text_utils import get_num_lines
import numpy as np


PROJECT_PATH = '/Users/satpathya/Dev/RNN_name_classifier/'

def set_project_path(project_path):

    global PROJECT_PATH
    if project_path[-1] != '/':
        project_path += '/'
    PROJECT_PATH = project_path


class NameDataset():

    def __init__(self,dir_path = 'Data/names/', percent_test = 0.20):

        if dir_path[0] =='/':
            dir_path = dir_path[1:]
        if dir_path[-1] != '/':
            dir_path += '/'
        self._dir_path = PROJECT_PATH + dir_path
        self._file_names = [f for f in listdir(self._dir_path) if isfile(join(self._dir_path,f))]
        self._classes = [f[:-4] for f in self._file_names]
        self._max_seq_length = None
        self._num_chars = None
        self._mode = 'TRAIN'
        self._len = 0
        self._i = 0
        self._char_to_idx = {}
        self._test_idx = {k:[] for k in self._classes}
        self._set_vars()
        self._train_test_split(percent_test)
        self._test_length = sum([len(self._test_idx[k]) for k in self._classes])
        self._train_length = self._len - self._test_length
        self._idx_char = {v:k for k,v in self._char_to_idx.items()}

    def get_classes(self):
        return self._classes

    def get_filenames(self):
        return self._file_names

    def get_num_classes(self):
        return len(self._classes)

    def get_max_seq_length(self):
        return self._max_seq_length

    def get_num_chars(self):
        return self._num_chars

    def get_char_id_map(self):
        return self._char_to_idx

    def get_id_char_map(self):
        return self._idx_char

    def get_test_ids(self):
        return self._test_idx

    def __len__(self):
        if self._mode == 'TRAIN':
            return self._train_length
        else:
            return self._test_length

    def _set_vars(self):
        self._max_seq_length = 0
        for f in self._file_names:
            filepath = self._dir_path + f
            with open(filepath, 'r') as file:
                for line in file:
                    self._len += 1
                    if len(line) > self._max_seq_length:
                        self._max_seq_length = len(line)
            file.close()

        for f in self._file_names:
            filepath = self._dir_path + f
            with open(filepath) as file:
                for line in file:
                    for c in line:
                        if c not in self._char_to_idx.keys():
                            self._char_to_idx[c] = len(self._char_to_idx)
            file.close()
        self._char_to_idx['EOW'] = len(self._char_to_idx)
        self._char_to_idx['SOW'] = len(self._char_to_idx)

        self._num_chars = len(self._char_to_idx)

    def _train_test_split(self, percent_test_ex):

        for filename in self._file_names:
            filepath = self._dir_path + filename
            with open(filepath) as file:
                num_lines = get_num_lines(filepath)
                num_text_ex = int(percent_test_ex*num_lines)
                self._test_idx[filename[:-4]] = np.random.choice(num_lines,num_text_ex)
            file.close()

    def __getitem__(self, item):

        if item > self._len:
            raise IndexError('Maximum limit reached - All examples covered')

        idx = self._i%len(self._classes)

        filepath = self._dir_path + self._file_names[idx]
        num_lines = get_num_lines(filepath)
        line_number = 0
        while True:
            line_number = np.random.choice(num_lines)
            if self._mode == 'TRAIN' and line_number not in self._test_idx[self._classes[idx]]:
                break
            if self._mode == 'TEST' and line_number in self._test_idx[self._classes[idx]]:
                break
        with open(filepath) as file:
            for index,line in enumerate(file):
                if index == line_number:
                    self._i += 1
                    return line,idx

    def flush_i(self):
        self._i = 0

    def toggle_mode(self,mode):
        assert mode in ['TRAIN','TEST'],'mode should either be TRAIN or TEST'
        self._mode = mode

















if __name__ == "__main__":

    Names = NameDataset(dir_path='/Data/names')
    print(Names.get_classes())
    print(Names.get_filenames())
    print(Names.get_char_id_map())
    print(Names.get_max_seq_length())
    print(Names.get_num_chars())
    print(len(Names.get_test_ids()))
    print(Names.get_num_chars())
    print(len(Names))

    for i in range(20):
        print(Names[i])
