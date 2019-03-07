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
        self.previous_visit = self.run_conf.get('Params').get('previous_visit')
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

            previous_visit = 6
            first = np.append(u_pos_input[0:1], u_pos_input[:-1], 0)
            second = np.append(u_pos_input[0:2], u_pos_input[:-2], 0)
            third = np.append(u_pos_input[0:3], u_pos_input[:-3], 0)
            fourth = np.append(u_pos_input[0:4], u_pos_input[:-4], 0)
            fifth = np.append(u_pos_input[0:5], u_pos_input[:-5], 0)
            six = np.append(u_pos_input[0:6], u_pos_input[:-6], 0)

            for counter in range(previous_visit):
                u_pos_input_all = np.append(u_pos_input[0:counter+1], u_pos_input[:-counter-1], 0)
            # print('u_pos_input_all', u_pos_input_all)

            # print('previous_visit', self.previous_visit)

            print('vessel_pathidx', vessel_pathidx)
            b = vessel_pathidx.copy()
            for i, src in enumerate(vessel_pathidx):
                b[i] = np.where(A[src, :])[0][0]
            b_pos_output = V[b, :]
            u_pos_input = u_pos_input.astype(np.int)
            maxoutput = 0.0
            maxout = 0.0
            maxphi = 0.0

            temp = img_obj.working_arr.copy()
            for (b, a), (d, c), (f, e), (h, g), (n, m), (q, p), (j, i), output, b_out in zip(six, fifth, fourth, third, second, first, u_pos_input, b_pos_output - u_pos_input, b_pos_output):
            # for (f, e), (h, g), (n, m), (q, p), (j, i), output, b_out in zip(fourth, third, second, first, u_pos_input, b_pos_output - u_pos_input, b_pos_output):

                temp[int(i), int(j)] = 255

                # for i, j in u_pos_input_all:
                #     row_from, row_to = int(i - self.k_half), int(i + self.k_half + 1)
                #     col_from, col_to = int(j - self.k_half), int(j + self.k_half + 1)
                #     if row_from < 0 or col_from < 0:
                #         continue
                #     if row_to >= img_obj.working_arr.shape[0] or col_to >= img_obj.working_arr.shape[1]:
                #         continue
                #     if np.isin(0, img_obj.mask[row_from:row_to, col_from:col_to]):
                #         continue
                #     u_pos_input_allofthem = np.append(u_pos_input_allofthem, i)
                #     u_pos_input_allofthem = np.append(u_pos_input_allofthem, j)
                #
                # print('u_pos_input_allofthem', u_pos_input_allofthem)

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

                row_from, row_to = int(m - self.k_half), int(m + self.k_half + 1)
                col_from, col_to = int(n - self.k_half), int(n + self.k_half + 1)
                if row_from < 0 or col_from < 0:
                    continue
                if row_to >= img_obj.working_arr.shape[0] or col_to >= img_obj.working_arr.shape[1]:
                    continue
                if np.isin(0, img_obj.mask[row_from:row_to, col_from:col_to]):
                    continue

                row_from, row_to = int(g - self.k_half), int(g + self.k_half + 1)
                col_from, col_to = int(h - self.k_half), int(h + self.k_half + 1)
                if row_from < 0 or col_from < 0:
                    continue
                if row_to >= img_obj.working_arr.shape[0] or col_to >= img_obj.working_arr.shape[1]:
                    continue
                if np.isin(0, img_obj.mask[row_from:row_to, col_from:col_to]):
                    continue

                row_from, row_to = int(e - self.k_half), int(e + self.k_half + 1)
                col_from, col_to = int(f - self.k_half), int(f + self.k_half + 1)
                if row_from < 0 or col_from < 0:
                    continue
                if row_to >= img_obj.working_arr.shape[0] or col_to >= img_obj.working_arr.shape[1]:
                    continue
                if np.isin(0, img_obj.mask[row_from:row_to, col_from:col_to]):
                    continue

                row_from, row_to = int(c - self.k_half), int(c + self.k_half + 1)
                col_from, col_to = int(d - self.k_half), int(d + self.k_half + 1)
                if row_from < 0 or col_from < 0:
                    continue
                if row_to >= img_obj.working_arr.shape[0] or col_to >= img_obj.working_arr.shape[1]:
                    continue
                if np.isin(0, img_obj.mask[row_from:row_to, col_from:col_to]):
                    continue

                row_from, row_to = int(a - self.k_half), int(a + self.k_half + 1)
                col_from, col_to = int(b - self.k_half), int(b + self.k_half + 1)
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

                #convert -pi-pi to 0-pi
                phi = abs(phi)
                # if phi < 0:
                    # phi = (2 * math.pi) + phi
                    # phi = math.pi + phi

                # convert pi to degree
                phi = phi * 180 / math.pi

                # convert 0-180 to 0-90
                if phi > 90:
                    phi = 180 - phi
                # if phi > 30:
                #     if phi < 165 or phi > 205:
                #         continue
                # # if phi > 15 :
                # #     continue


                # print('iiijjj', output[0], output[1])
                # phi = abs(np.dot([1, 0], [output[0], output[1]]))

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
                if phi > maxphi:
                    maxphi = phi
                # print('iiii,,,jjjj', i, j)

                self.indices.append([ID, [a, b], [c, d], [e, f], [g, h], [m, n], [p, q], [i, j], phi, output])
                # self.indices.append([ID, [e, f], [g, h], [m, n], [p, q], [i, j], phi, output])
            IMG.fromarray(temp).save('patches/'+img_file+'.png')
            # print('maxxiiimuum', maxout, maxoutput)
            # print('maxphi', maxphi)
            # print('getitem', self.indices[0])

    def __getitem__(self, index):
        ID, (a, b), (c, d), (e, f), (g, h), (m, n), (p, q), (i, j), out, b_pos = self.indices[index]

        row_from, row_to = a - self.k_half, a + self.k_half + 1
        col_from, col_to = b - self.k_half, b + self.k_half + 1

        row_from = int(row_from)
        row_to = int(row_to)
        col_from = int(col_from)
        col_to = int(col_to)
        sixth_prev_patches = self.image_objects[ID].res['2d'][:, row_from:row_to, col_from:col_to]

        row_from, row_to = c - self.k_half, c + self.k_half + 1
        col_from, col_to = d - self.k_half, d + self.k_half + 1

        row_from = int(row_from)
        row_to = int(row_to)
        col_from = int(col_from)
        col_to = int(col_to)
        fifth_prev_patches = self.image_objects[ID].res['2d'][:, row_from:row_to, col_from:col_to]

        row_from, row_to = e - self.k_half, e + self.k_half + 1
        col_from, col_to = f - self.k_half, f + self.k_half + 1

        row_from = int(row_from)
        row_to = int(row_to)
        col_from = int(col_from)
        col_to = int(col_to)
        fourth_prev_patches = self.image_objects[ID].res['2d'][:, row_from:row_to, col_from:col_to]

        row_from, row_to = g - self.k_half, g + self.k_half + 1
        col_from, col_to = h - self.k_half, h + self.k_half + 1

        row_from = int(row_from)
        row_to = int(row_to)
        col_from = int(col_from)
        col_to = int(col_to)
        third_prev_patches = self.image_objects[ID].res['2d'][:, row_from:row_to, col_from:col_to]

        row_from, row_to = m - self.k_half, m + self.k_half + 1
        col_from, col_to = n - self.k_half, n + self.k_half + 1

        row_from = int(row_from)
        row_to = int(row_to)
        col_from = int(col_from)
        col_to = int(col_to)
        sec_prev_patches = self.image_objects[ID].res['2d'][:, row_from:row_to, col_from:col_to]

        row_from, row_to = p - self.k_half, p + self.k_half + 1
        col_from, col_to = q - self.k_half, q + self.k_half + 1

        row_from = int(row_from)
        row_to = int(row_to)
        col_from = int(col_from)
        col_to = int(col_to)
        first_prev_patches = self.image_objects[ID].res['2d'][:, row_from:row_to, col_from:col_to]

        row_from, row_to = i - self.k_half, i + self.k_half + 1
        col_from, col_to = j - self.k_half, j + self.k_half + 1

        row_from = int(row_from)
        row_to = int(row_to)
        col_from = int(col_from)
        col_to = int(col_to)
        input_patches = self.image_objects[ID].res['2d'][:, row_from:row_to, col_from:col_to]
        # input_patches = np.where(input_patches > 135, 1, 0)

        # input_tensor = np.array([sixth_prev_patches, fifth_prev_patches, fourth_prev_patches, third_prev_patches, sec_prev_patches, first_prev_patches, input_patches])
        input_tensor_temp0 = np.append(sixth_prev_patches, fifth_prev_patches, 0)
        input_tensor_temp1 = np.append(input_tensor_temp0, fourth_prev_patches, 0)
        input_tensor_temp2 = np.append(input_tensor_temp1, third_prev_patches, 0)
        input_tensor_temp3 = np.append(input_tensor_temp2, sec_prev_patches, 0)
        input_tensor_temp4 = np.append(input_tensor_temp3, first_prev_patches, 0)
        input_tensor = np.append(input_tensor_temp4, input_patches, 0)
        # input_tensor = np.where(input_tensor > 135, 1, 0)
        # if out == 0:
        #     print('hhhss', out, a, b, c, d, e, f, g, h, m, n, p, q, i, j, b_pos)
        # print('valuuueeeesss', input_tensor.shape)
        # print('self.indices', self.indices)
            # IMG.fromarray(img_obj)
            # IMG.fromarray(input_tensor[12, :, :]).save('patches/'+str(index) + '.png')
            # IMG.fromarray(input_tensor[13, :, :]).save('patches/'+ str(index) + '_1.png')

        return {'IDs': ID, 'POS': np.array([i, j]), 'PREV': np.array([p, q]), 'inputs': input_tensor,
                'labels': torch.FloatTensor([out])}
