import os

import numpy as np
import torch

from neuralnet.datagen import Generator
import math
from commons.MAT import Mat
from commons.IMAGE import Image
sep = os.sep


class PatchesGenerator(Generator):
    def __init__(self, **kwargs):
        super(PatchesGenerator, self).__init__(**kwargs)
        self.patch_shape = self.run_conf.get('Params').get('patch_shape')
        self.patch_pad = self.run_conf.get('Params').get('patch_pad')
        self.patch_offset = self.run_conf.get('Params').get('patch_offset')
        self.previous_visit = self.run_conf.get('Params').get('previous_visit')
        self.k_half = int(math.floor(self.patch_shape[0] / 2))
        self._load_indices()
        print('Patches:', self.__len__())

    def _load_indices(self):

        def _is_valid(i, j):
            row_from, row_to = int(i - self.k_half), int(i + self.k_half + 1)
            col_from, col_to = int(j - self.k_half), int(j + self.k_half + 1)
            if row_from < 0 or col_from < 0:
                return False
            if row_to >= img_obj.working_arr.shape[0] or col_to >= img_obj.working_arr.shape[1]:
                return False
            if np.isin(0, img_obj.mask[row_from:row_to, col_from:col_to]):
                return False
            return True

        for ID, img_file in enumerate(self.images):
            mat_file = Mat(mat_file=self.image_dir + sep + img_file)

            V = mat_file.get_graph('V')
            A = mat_file.get_graph('A')
            I = mat_file.get_image('I')
            T = np.max(mat_file.get_image('T'), 2)

            img_obj = Image()
            img_obj.file_name = img_file
            img_obj.image_arr = I
            img_obj.working_arr = I[:, :, 1]
            img_obj.res['2d'] = np.array([T, 255 - I[:, :, 1]])

            img_obj.load_mask(self.mask_dir, self.mask_getter)

            self.image_objects[ID] = img_obj

            path_index = mat_file.get_graph('pathNode')
            vessel_pathidx = np.where(path_index == 1)[0]
            u_pos_input = V[vessel_pathidx, :]

            positions = [u_pos_input]
            for ix in range(1, self.previous_visit + 1):
                prev = np.append(u_pos_input[0:ix], u_pos_input[:-ix], 0)
                positions.append(prev)

            # print('positions', np.shape(positions))
            print('vessel_pathidx', vessel_pathidx)
            b = vessel_pathidx.copy()
            for ix, src in enumerate(vessel_pathidx):
                b[ix] = np.where(A[src, :])[0][0]
            b_pos_output = V[b, :]
            u_pos_input = u_pos_input.astype(np.int)

            relative_xy = b_pos_output - u_pos_input

            for ix in range(u_pos_input.shape[0]):
                indices = []
                for pos in positions:
                    xy = pos[ix][::-1]
                    if _is_valid(xy[0], xy[1]):
                        indices.append(xy)
                if len(indices) - 1 == self.previous_visit:
                    self.indices.append([ID, indices, relative_xy[ix]])
        if self.mode == 'train':
            import matplotlib.pyplot as plt
            plt.rcParams["figure.figsize"] = (12, 8)
            # plt.plot(np.arange(361), list(angcount.values()))
            plt.grid(True)
            # plt.hist(hist_data, bins=18)
            # plt.savefig('patch_distrib_fig1.png')

    def __getitem__(self, index):
        ID, positions, rel_out = self.indices[index]

        input_tensor = []
        for ix, (a, b) in enumerate(positions):
            row_from, row_to = a - self.k_half, a + self.k_half + 1
            col_from, col_to = b - self.k_half, b + self.k_half + 1
            row_from = int(row_from)
            row_to = int(row_to)
            col_from = int(col_from)
            col_to = int(col_to)
            _patch = self.image_objects[ID].res['2d'][:, row_from:row_to, col_from:col_to]
            input_tensor.append(_patch[0])
            input_tensor.append(_patch[1])

        input_tensor = np.array(input_tensor)
        return {'IDs': ID, 'POS': np.array(positions[0]), 'inputs': input_tensor,
                'labels': torch.FloatTensor([rel_out])}

