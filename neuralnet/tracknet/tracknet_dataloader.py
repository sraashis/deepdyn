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
            img_obj.res['2d'] = np.array([T, 255 - I[:, :, 1]])

            img_obj.load_mask(self.mask_dir, self.mask_getter)

            self.image_objects[ID] = img_obj

            path_index = mat_file.get_graph('pathNode')
            vessel_pathidx = np.where(path_index == 1)[0]
            u_pos_input = V[vessel_pathidx, :]
            u_pos_input_prev = np.append(u_pos_input[0][None, ...], u_pos_input[:-1], 0)

            print('vessel_pathidx', vessel_pathidx)
            b = vessel_pathidx.copy()
            for i, src in enumerate(vessel_pathidx):
                b[i] = np.where(A[src, :])[0][0]
            b_pos_output = V[b, :]
            u_pos_input = u_pos_input.astype(np.int)
            maxoutput = 0.0
            maxout = 0.0

            temp = img_obj.working_arr.copy()
            for (q, p), (j, i), output, b_out in zip(u_pos_input_prev, u_pos_input, b_pos_output - u_pos_input, b_pos_output):
                row_from, row_to = int(i - self.k_half), int(i + self.k_half + 1)
                col_from, col_to = int(j - self.k_half), int(j + self.k_half + 1)
                if row_from < 0 or col_from < 0:
                    continue
                if row_to >= img_obj.working_arr.shape[0] or col_to >= img_obj.working_arr.shape[1]:
                    continue
                if np.isin(0, img_obj.mask[row_from:row_to, col_from:col_to]):
                    continue

                temp[int(i), int(j)] = 255
                row_from, ro1w_to = int(p - self.k_half), int(p + self.k_half + 1)
                col_from, col_to = int(q - self.k_half), int(q + self.k_half + 1)
                if row_from < 0 or col_from < 0:
                    continue
                if row_to >= img_obj.working_arr.shape[0] or col_to >= img_obj.working_arr.shape[1]:
                    continue
                if np.isin(0, img_obj.mask[row_from:row_to, col_from:col_to]):
                    continue

                # print('output before', output)
                # your_permutation = [1, 0]
                # order = np.argsort(your_permutation)
                # output[:, order]
                # print('ooouutt', output)

                rho = np.sqrt(output[0] ** 2 + output[1] ** 2)
                phi = np.arctan2(output[1], output[0])
                # print(i, j, b_out, phi)

                if phi < 0:
                    phi = (2 * math.pi) + phi
                    # phi = math.pi + phi

                phi = phi * 180 / math.pi

                # if phi > (math.pi / 2):
                #         phi = math.pi - phi
                # print('ii,jj, b_out', i, j,  b_out)

                # if phi == 0:
                #     img_obj = 255
                # if maxout < abs(output[0]):
                #     maxout = abs(output[0])
                # if maxoutput < abs(output[1]):
                #     maxoutput = abs(output[1])
                # s = s[::-1]
                b_out = b_out[::-1]
                # print('i, j, output', i, j, p, q, type(b_out), b_out)
                # print('phi', phi)
                self.indices.append([ID, [p, q], [i, j], phi, output])
            IMG.fromarray(temp).save('patches/'+img_file+'.png')
            # print('maxxiiimuum', maxout, maxoutput)

    def __getitem__(self, index):
        ID, (p, q), (i, j), out, b_pos = self.indices[index]

        row_from, row_to = p - self.k_half, p + self.k_half + 1
        col_from, col_to = q - self.k_half, q + self.k_half + 1

        row_from = int(row_from)
        row_to = int(row_to)
        col_from = int(col_from)
        col_to = int(col_to)
        prev_patches = self.image_objects[ID].res['2d'][:, row_from:row_to, col_from:col_to]

        row_from, row_to = i - self.k_half, i + self.k_half + 1
        col_from, col_to = j - self.k_half, j + self.k_half + 1

        row_from = int(row_from)
        row_to = int(row_to)
        col_from = int(col_from)
        col_to = int(col_to)
        input_patches = self.image_objects[ID].res['2d'][:, row_from:row_to, col_from:col_to]

        input_tensor = np.append(prev_patches, input_patches, 0)
        # if out > 350:
        #     print('hhhss', out, i, j, b_pos)
            # IMG.fromarray(img_obj)
            # IMG.fromarray(input_patches[1, :, :]).save('patches/'+str(index) + '.png')
        return {'IDs': ID, 'POS': np.array([i, j]), 'PREV': np.array([p, q]), 'inputs': input_tensor,
                'labels': torch.FloatTensor([out])}
