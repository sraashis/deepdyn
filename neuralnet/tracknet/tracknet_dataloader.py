import os

import numpy as np
import torch

from neuralnet.datagen import Generator
import math
from commons.MAT import Mat
from commons.IMAGE import Image
from PIL import Image as IMG

sep = os.sep


class PatchesGenerator(Generator):
    def __init__(self, **kwargs):
        super(PatchesGenerator, self).__init__(**kwargs)
        self.patch_shape = self.run_conf.get('Params').get('patch_shape')
        self.patch_pad = self.run_conf.get('Params').get('patch_pad')
        self.patch_offset = self.run_conf.get('Params').get('patch_offset')
        self.k_half = int(math.floor(self.patch_shape[0] / 2))
        self._load_indices()
        print('Patches:', self.__len__())

    def _load_indices(self):
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
            img_obj.res['2d'] = np.array([T, 255-I[:, :, 1]])




            img_obj.load_mask(self.mask_dir, self.mask_getter)

            self.image_objects[ID] = img_obj

            path_index = mat_file.get_graph('pathNode')
            vessel_pathidx = np.where(path_index == 1)[0]
            u_pos_input = V[vessel_pathidx, :]
            u_pos_input_prev = np.append(u_pos_input[0][None, ...], u_pos_input[:-1], 0)

            print('vessel_pathidx', vessel_pathidx)
            b = vessel_pathidx.copy()
            # a = vessel_pathidx.copy()
            for i, src in enumerate(vessel_pathidx):
                b[i] = np.where(A[src, :])[0][0]
            b_pos_output = V[b, :]

            # for i, src in enumerate(vessel_pathidx):
            #     a[i] = np.where(A[:, src])[0][0]
            # a_pos_output = V[a, :]

            u_pos_input = u_pos_input.astype(np.int)

            # we need the following line for x, y coordinate
            # b_pos_output = b_pos_output.astype(np.int)
            for (p, q), (i, j), output in zip(u_pos_input_prev, u_pos_input, b_pos_output - u_pos_input):
                row_from, row_to = int(i - self.k_half), int(i + self.k_half + 1)
                col_from, col_to = int(j - self.k_half), int(j + self.k_half + 1)
                if row_from < 0 or col_from < 0:
                    continue
                if row_to >= img_obj.working_arr.shape[0] or col_to >= img_obj.working_arr.shape[1]:
                    continue
                if np.isin(0, img_obj.mask[row_from:row_to, col_from:col_to]):
                    continue

                row_from, row_to = int(p - self.k_half), int(p + self.k_half + 1)
                col_from, col_to = int(q - self.k_half), int(q + self.k_half + 1)
                if row_from < 0 or col_from < 0:
                    continue
                if row_to >= img_obj.working_arr.shape[0] or col_to >= img_obj.working_arr.shape[1]:
                    continue
                if np.isin(0, img_obj.mask[row_from:row_to, col_from:col_to]):
                    continue

                print('output[0], output[1])', output[0], output[1])
                rho = np.sqrt(output[0] ** 2 + output[1] ** 2)
                # phi = math.degrees(np.arctan2(output[0], output[1]))
                # print('maxx', np.max(phi), np.min(phi))
                phi = np.arctan2(output[0], output[1])
                self.indices.append([ID, [p, q], [i, j], [rho, phi]])
                # self.indices.append([ID, [p, q], [i, j], output.tolist()])

    def __getitem__(self, index):
        ID, (p, q), (i, j), out = self.indices[index]

        row_from, row_to = p - self.k_half, p + self.k_half + 1
        col_from, col_to = q - self.k_half, q + self.k_half + 1

        row_from = int(row_from)
        row_to = int(row_to)
        col_from = int(col_from)
        col_to = int(col_to)
        prev_patches = self.image_objects[ID].res['2d'][:, row_from:row_to, col_from:col_to]
        # Simplification 1: Make the target pixel white
        # try:
        #     prev_patches[out[0], out[1], 0] = 255
        # except:
        #     out = [0, 0]
        #     prev_patches[out[0], out[1], 0] = 255

        row_from, row_to = i - self.k_half, i + self.k_half + 1
        col_from, col_to = j - self.k_half, j + self.k_half + 1

        row_from = int(row_from)
        row_to = int(row_to)
        col_from = int(col_from)
        col_to = int(col_to)
        input_patches = self.image_objects[ID].res['2d'][:, row_from:row_to, col_from:col_to]
        # print(input_patches.shape)
        # print('oooouuuutttt', out)
        # try:
        #     input_patches[:, out[0], out[1]] = 255
        # except:
        #     out = [0, 0]
        #     input_patches[:, out[0], out[1]] = 255
        # print('outtt', out)
        # print('input_patches[out[0], out[1], 0] ', input_patches[out[0], out[1], 0] )

        input_tensor = np.append(prev_patches, input_patches, 0)
        return {'IDs': ID, 'POS': np.array([i, j]), 'PREV': np.array([p,q]), 'inputs': input_tensor,
                'labels': torch.FloatTensor(out)}