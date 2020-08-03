def density_adaptive_kernel(data_root  = './train', min_sigma = 2, data_limit=None, showden = False):
    # min_sigma can be set as 0

    #img_pathes = glob.glob(f'{data_root}*/images/*.jpg')
    # for img_path in tqdm(img_pathes[:data_limit]):
    #     data_folder, img_sub_path = img_path.split('images')
    #     #print("The path for data is: ", data_folder)
    #     #print("The path for image is: ", img_path)
    #     ann_path = img_path.replace(img_affix, '.txt')
    #     ann_path = ann_path.replace("images", 'annotations')
    #     #print("The file path: ", ann_path)
    #     #txt_data = open(ann_path, 'r').readlines()
    #
    #     # load img and map
    #     img = Image.open(img_path)
    #     width, height = img.size
    #     gt_points = get_gt_dots(txt_data, height, width, mode=mode)
    #
    #     distances = distances_dict[img_path]
    #     density_map = gaussian_filter_density(gt_points, height, width, distances, kernels_dict,
#                                                                       min_sigma=min_sigma, method=1)
    #     if showden:
        #     plt.imshow(img)
        #     plt.imshow(density_map, alpha=0.75)
        #     plt.show()
    #    else:
    #    plt.imshow(density_map)
    #    plt.savefig(den_name, format="jpg")
    return