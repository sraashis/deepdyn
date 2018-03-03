from commons.IMAGE import Image
import preprocess.av.image_filters as fil

if __name__ == '__main__':
    img = Image('wide_image_13.mat')
    # img.show_image(image_array=img.img_array)
    img.load_kernel_bank()
    img.apply_bilateral(img.image_arr[:, :, 1])
    # img.show_image(img.img_bilateral)
    img_temp = img.image_arr[:, :, 1] - img.img_bilateral
    img.apply_gabor(arr=img_temp, filter_bank=fil.get_chosen_gabor_bank())
    # img.histogram(255 - img.img_gabor)
    # img.img_gabor[img.img_gabor<250] = 0
    img.show_image(255-img.img_gabor)
    # img.show_image(img.img_array)
    # img.show_kernel(fil.get_chosen_gabor_bank())
    # img.show_av_graph(mat_file=img.mat, image_arr=img.img_gabor, av_only=False)
