# Plume vetting paper: https://www.sciencedirect.com/science/article/pii/S0034425725002640
# Identification of false methane plumes for orbital imaging spectrometers: A case study with EMIT
# Chuchu Xiang, David R. Thompson, Robert O. Green, Jay E. Fahlen, Andrew K. Thorpe, Philip G. Brodrick, Red Willow Coleman, Amanda M. Lopez, Clayton D. Elder

from .plume_vetting_sfun import get_radiance_ratio, calculate_fit, calculate_dist, calculate_magnitude, find_uniform_indices
import numpy as np
import matplotlib.pyplot as plt
from georeader.readers.emit import EMITImage
from georeader.geotensor import GeoTensor
from shapely.geometry import Polygon
from shapely.ops import unary_union
from georeader import rasterize, read, window_utils


def how_many_pixels_does_polygon_occupy(polygon, ref_data:GeoTensor):
    # potentially external dependencies ...
    
    rasterized = rasterize.rasterize_geometry_like(polygon, data_like=ref_data, value=1, fill=0, crs_geometry="EPSG:4326")
    number_of_pixels = np.sum(rasterized.values)
    return number_of_pixels, rasterized.values


def compute_other_plumes_exclusion_mask(
    vectors: list[Polygon], pol_idx_target: int, ref_data: GeoTensor
) -> GeoTensor:
    """Rasterize non-target polygons into a mask aligned to ``ref_data``."""
    other_polygons = [
        polygon_other
        for pol_idx_other, polygon_other in enumerate(vectors)
        if pol_idx_other != pol_idx_target
    ]
    if len(other_polygons) == 0:
        return GeoTensor(
            np.zeros_like(ref_data.values, dtype=bool),
            transform=ref_data.transform,
            crs=ref_data.crs,
            fill_value_default=False,
        )

    union_other_polygons = unary_union(other_polygons)
    return rasterize.rasterize_geometry_like(
        union_other_polygons,
        data_like=ref_data,
        value=1,
        fill=0,
        crs_geometry="EPSG:4326",
    )

def compute_masks(mf, mf_threshold=30, clouds_and_surface_water_mask=None, debug=False):
    ######### calculate bad pixel mask ################
    uni_row, uni_col = find_uniform_indices(mf)  # find rows and columns with two or fewer unique MF values
    rowcol_mask = np.full_like(clouds_and_surface_water_mask, False, dtype=bool)
    rowcol_mask[uni_row, :] = True
    rowcol_mask[:, uni_col] = True
    rowcol_mask[mf <= -9999] = True

    # This mask is used for target pixels. Combine the rowcol_mask with the clouds_and_surface_water_mask
    combined_mask = clouds_and_surface_water_mask | rowcol_mask

    # This mask is used for background pixels
    background_mask = combined_mask.copy()
    background_mask[(mf < -mf_threshold) | (mf > mf_threshold)] = True

    if debug:
        plt.imshow(combined_mask)
        plt.title("combined_mask")
        plt.show()
        plt.imshow(background_mask)
        plt.title("background_mask")
        plt.show()

    return combined_mask, background_mask

def coords_inside_plume_from_binary(thresholded_instance, debug=False):
    points_inside_plume = []
    minx, miny, maxx, maxy = 0, 0, thresholded_instance.shape[0], thresholded_instance.shape[1]
    for y in range(int(miny), int(maxy)):
        for x in range(int(minx), int(maxx)):
            if thresholded_instance[x,y]:
                points_inside_plume.append((y, x))

    if debug:
        debug_viz_data = np.zeros_like(thresholded_instance)
        for coords in points_inside_plume:
            y, x = coords
            debug_viz_data[x, y] = 1.
        plt.imshow(debug_viz_data)
        plt.show()

    return points_inside_plume

# === scene specific target signature ===
# TODO: REPLACE WITH YOUR OWN CALL TO target_generation
SCENE_TARGET_SIGNATURE_DEFAULT = np.asarray([9.65677880e-06, 9.44929334e-06, 9.04269344e-06, 8.85827101e-06, 8.98958035e-06, 8.49728372e-06, 8.62805714e-06, 8.80114715e-06, 8.33883158e-06, 8.16142819e-06, 7.73368404e-06, 7.39366692e-06, 7.35562982e-06, 7.04368029e-06, 6.11562144e-06, 6.53218762e-06, 6.63357218e-06, 5.78389142e-06, 5.84881855e-06, 5.47753266e-06, 4.79791781e-06, 1.13047456e-06, -2.35286791e-06, 4.59439270e-06, 4.56166689e-06, 3.33127292e-06, 1.54112465e-06, 3.26023549e-06, 3.48084766e-06, 2.07768389e-06, 1.34280623e-06, -1.43228457e-05, -5.30431391e-05, -1.43933273e-05, 1.97451749e-06, 2.04592925e-06, 1.06639434e-06, -5.74711749e-06, -1.36636143e-05, -8.58726232e-06, -5.89627356e-07, -2.22065616e-06, -3.23409978e-06, -2.62903649e-05, -2.16494465e-05, -5.92926997e-05, -2.94537249e-04, -3.05365354e-04, -6.61797152e-05, -3.59345933e-06, -9.58845533e-08, -6.98856370e-06, -2.20578091e-05, -4.63919870e-05, -9.98187735e-05, -1.06221661e-04, -1.34282159e-04, -8.69059866e-05, -3.41956800e-05,
                    -7.52936476e-06, -1.03710475e-06, -4.22974056e-05, -8.64636580e-05, -7.36054663e-05, -3.05240747e-04, -5.80721283e-04, -5.79022997e-04, -3.15508683e-03, -7.20142290e-03, -5.69085827e-03, -2.20352407e-03, -4.58517744e-04, -1.39371442e-04, -3.95658213e-05, -1.05974738e-05, -6.18653290e-06, -3.43563370e-05, -6.65568568e-05, -2.57255106e-04, -5.96432718e-04, -6.37708274e-04, -1.44363145e-03, -1.96407999e-03, -1.44413259e-03, -1.86983579e-03, -1.77950763e-03, -7.16888202e-04, -4.00458836e-04, -2.25835846e-04, -1.02706943e-04, -2.08844006e-05, -1.18141154e-07, 5.29685862e-07, -1.92552801e-07, -6.41317179e-06, -1.58942158e-04, -2.15882719e-04, -4.84205933e-04, -2.26379462e-03, -6.85842307e-03, -1.50126939e-02, -1.57474495e-02, -1.18301261e-02, -2.05601913e-02, -3.60834425e-02, -3.94273630e-02, -2.59847522e-02, -1.44271172e-02, -9.88838744e-03, -6.08667109e-03, -3.18633724e-03, -1.46129171e-03, -5.25151565e-04, -2.60101417e-04, -3.37112286e-04, -1.97430909e-04,
                    -8.00361556e-05, -1.10455109e-05, -1.31887203e-05, -4.19767046e-05, -3.32164105e-05, -2.49744551e-05, -7.15452259e-05, -3.87402691e-04, -2.02328613e-03, -7.70922092e-03, -1.23149560e-02, -2.02294852e-02, -2.76472263e-02, -2.87585064e-02, -2.83417192e-02, -3.16582578e-02, -2.93610347e-02, -3.23032725e-02, -3.81929431e-02, -3.90207197e-02, -3.29549597e-02, -3.48682336e-02, -3.12618589e-02, -2.05089329e-02, -1.49367305e-02, -1.07353593e-02, -5.97769992e-03, -4.02974442e-03, -2.81336234e-03, -2.08011794e-03, -2.54418498e-03, -2.72301685e-03, -1.72132735e-03, -9.38626716e-04, -5.69493323e-04, -4.56683372e-04, -3.41620643e-04, -2.17749254e-04, -1.29124440e-04, -9.29762234e-05, -6.92330156e-05, -5.67021771e-05, -6.96139050e-05, -6.38932674e-05, -4.11656578e-05, -3.67605277e-05, -7.57429864e-05, -2.71394567e-04, -1.01682139e-03, -3.75419836e-03, -1.26479782e-02, -3.75491237e-02, -8.38793133e-02, -1.13786173e-01, -1.16849649e-01, -8.59692033e-02, -1.38584499e-01,
                    -1.44295087e-01, -6.46457727e-02, -7.58295710e-02, -8.64002790e-02, -1.05684196e-01, -1.23224796e-01, -6.18103379e-02, -8.05863781e-02, -1.00805092e-01, -6.55649471e-02, -4.37739533e-02, -3.14234067e-02, -3.00320311e-02, -4.31862871e-02, -6.02494192e-02, -4.14973363e-02, -7.14254345e-02, -5.12582042e-02, -2.41301424e-02, -2.49192260e-02, -2.28203546e-02, -1.85601865e-02, -1.44834816e-02, -1.14326348e-02, -8.08636595e-03, -6.13045088e-03, -4.44505264e-03, -2.75363038e-03, -1.76782748e-03, -1.29662555e-03, -1.13509987e-03, -1.27273834e-03, -1.68181124e-03, -2.11492102e-03, -2.41104042e-03, -1.42739322e-03, -9.57903434e-04, -1.20352719e-03, -1.00580261e-03, -6.22048310e-04, -6.37473072e-04, -5.43213916e-04, -5.01025722e-04, -3.52837832e-04, -2.48386828e-04, -2.06771481e-04, -1.71915054e-04, -1.75783825e-04, -9.43188133e-05, -7.43165031e-05, -4.79766004e-05, -4.43891285e-05, -2.59220423e-05, -2.82430025e-05, -4.60777162e-05, -6.02059693e-05, -6.53891201e-05,
                    -9.07074440e-05, -1.41093437e-04, -2.96758124e-04, -7.87594072e-04, -2.17192119e-03, -5.58628668e-03, -1.28307233e-02, -2.48689824e-02, -4.44926900e-02, -6.73228909e-02, -7.92775824e-02, -7.57375261e-02, -5.13581699e-02, -3.93471604e-02, -2.78321330e-01, -3.55763275e-01, -9.97541568e-02, -1.58274724e-01, -2.33960939e-01, -3.49164500e-01, -4.32103636e-01, -5.05408477e-01, -5.59860848e-01, -6.24071005e-01, -5.76580213e-01, -5.90172912e-01, -6.55077760e-01, -8.45815529e-01, -9.15431133e-01, -5.23962589e-01, -7.43168727e-01, -7.78706093e-01, -5.71164846e-01, -8.73710217e-01, -1.14039687e+00, -1.03317436e+00, -6.33479317e-01, -8.39119315e-01, -1.10115544e+00, -5.98434457e-01, -5.94363617e-01, -5.53028991e-01, -4.19743280e-01, -2.49515472e-01, -2.23981385e-01, -2.46002450e-01, -1.57568885e-01, -1.20763738e-01, -1.00731546e-01, -7.19952086e-02, -4.84661609e-02, -3.50186769e-02, -2.38227621e-02, -2.20361438e-02, -1.95619810e-02])

def compute(ei:EMITImage, cmf:GeoTensor, vectors:list[Polygon],
            scene_target_signature:np.ndarray=None, debug_viz:bool = True,
            num_pts:int = 40, min_polygon_size:int = 0, debug_jump_to:int = None) -> dict[dict[str, float]]:
    
    if scene_target_signature is None:
        scene_target_signature = SCENE_TARGET_SIGNATURE_DEFAULT.copy()
    
    mask_raw = np.array(ei.nc_ds_l2amask["mask"])
    clouds_and_surface_water_mask = np.sum(mask_raw[..., :3], axis=-1) > 0
    clouds_and_surface_water_mask_geo = ei.georreference(clouds_and_surface_water_mask, fill_value_default=True)

    data = ei.load(as_reflectance=False)

    rdn_ = data.values
    rdn_ = np.transpose(rdn_, axes=(1, 2, 0)) # Ch, W, H => i want into W,H,Ch again
    rdn_masked = np.where(rdn_ == -9999, 0, rdn_) # masked with 0 is close to how it's loaded in the main script ...
    rdn = rdn_masked

    # TODO units must be ppm x m
    cmf = cmf.copy()
    cmf.values[cmf.values == cmf.fill_value_default] = 0
    cmf.fill_value_default = 0.0

    if not cmf.same_extent(data):
        cmf = read.read_reproject_like(cmf, data, fill_value_default=0.0)

    # HYPERPARAMS:
    mf_threshold = 30 # The range of matched filter values around zero for the background pixels
    radius = 200 # how far from the plume center do we look for points?
    deg_poly = 10  # degree of polynomial for the theoretical model # (paper maybe mentioned 6?)

    # print("mask",mask.shape)


    scores_per_polygon = {}
    for pol_idx, polygon in enumerate(vectors):
        if debug_jump_to is not None:
            # DEBUG ... zoom into just one:
            if pol_idx != debug_jump_to: continue

        number_of_pixels, plume_mask = how_many_pixels_does_polygon_occupy(polygon, cmf)
        # only consider larger plumes?
        if number_of_pixels > min_polygon_size:
            title = "Polygon #"+str(pol_idx)+" ("+str(number_of_pixels)+"px)"
            print(title)

            combined_mask, background_mask = compute_masks(cmf.values, mf_threshold, clouds_and_surface_water_mask=clouds_and_surface_water_mask_geo.values, debug=False)
            exclusion_mask = compute_other_plumes_exclusion_mask(vectors, pol_idx, cmf)
            background_mask = background_mask | exclusion_mask.values.astype(bool)
            orig_points_inside_plume = coords_inside_plume_from_binary(plume_mask, False)

            # TODO: probably can be easily read from the data, I wanted to check them here super visibly when debugging ... feel free to make this more elegant :)!
            # wavelenghts in EMIT file (can be also loaded from the image ...)
            # emit_image.wavelengths
            # wl_orig = [381.00558, 388.4092, 395.81583, 403.2254, 410.638, 418.0536, 425.47214, 432.8927, 440.31726, 447.7428, 455.17035, 462.59888, 470.0304, 477.46292, 484.89743, 492.33292, 499.77142, 507.2099, 514.6504, 522.0909, 529.5333, 536.9768, 544.42126, 551.8667, 559.3142, 566.7616, 574.20905, 581.6585, 589.108, 596.55835, 604.0098, 611.4622, 618.9146, 626.36804, 633.8215, 641.2759, 648.7303, 656.1857, 663.6411, 671.09753, 678.5539, 686.0103, 693.4677, 700.9251, 708.38354, 715.84094, 723.2993, 730.7587, 738.2171, 745.6765, 753.1359, 760.5963, 768.0557, 775.5161, 782.97754, 790.4379, 797.89935, 805.36176, 812.8232, 820.2846, 827.746, 835.2074, 842.66986, 850.1313, 857.5937, 865.0551, 872.5176, 879.98004, 887.44147, 894.90393, 902.3664, 909.82886, 917.2913, 924.7538, 932.21625, 939.6788, 947.14026, 954.6027, 962.0643, 969.5268, 976.9883, 984.4498, 991.9114, 999.37286, 1006.8344, 1014.295, 1021.7566, 1029.2172, 1036.6777, 1044.1383, 1051.5989, 1059.0596, 1066.5201, 1073.9797,
            #            1081.4404, 1088.9, 1096.3597, 1103.8184, 1111.2781, 1118.7368, 1126.1964, 1133.6552, 1141.1129, 1148.5717, 1156.0304, 1163.4882, 1170.9459, 1178.4037, 1185.8616, 1193.3184, 1200.7761, 1208.233, 1215.6898, 1223.1467, 1230.6036, 1238.0596, 1245.5154, 1252.9724, 1260.4283, 1267.8833, 1275.3392, 1282.7942, 1290.2502, 1297.7052, 1305.1603, 1312.6144, 1320.0685, 1327.5225, 1334.9756, 1342.4287, 1349.8818, 1357.3351, 1364.7872, 1372.2384, 1379.6907, 1387.1418, 1394.5931, 1402.0433, 1409.4937, 1416.944, 1424.3933, 1431.8427, 1439.292, 1446.7404, 1454.1888, 1461.6372, 1469.0847, 1476.5321, 1483.9796, 1491.4261, 1498.8727, 1506.3192, 1513.7649, 1521.2104, 1528.655, 1536.1007, 1543.5454, 1550.9891, 1558.4329, 1565.8766, 1573.3193, 1580.7621, 1588.205, 1595.6467, 1603.0886, 1610.5295, 1617.9705, 1625.4104, 1632.8513, 1640.2903, 1647.7303, 1655.1694, 1662.6074, 1670.0455, 1677.4836, 1684.9209, 1692.358, 1699.7952, 1707.2314, 1714.6667, 1722.103, 1729.5383, 1736.9727, 1744.4071,
            #            1751.8414, 1759.2749, 1766.7084, 1774.1418, 1781.5743, 1789.007, 1796.4385, 1803.8701, 1811.3008, 1818.7314, 1826.1611, 1833.591, 1841.0206, 1848.4495, 1855.8773, 1863.3052, 1870.733, 1878.16, 1885.5869, 1893.013, 1900.439, 1907.864, 1915.2892, 1922.7133, 1930.1375, 1937.5607, 1944.9839, 1952.4071, 1959.8295, 1967.2518, 1974.6732, 1982.0946, 1989.515, 1996.9355, 2004.355, 2011.7745, 2019.1931, 2026.6118, 2034.0304, 2041.4471, 2048.865, 2056.2808, 2063.6965, 2071.1123, 2078.5273, 2085.9421, 2093.3562, 2100.769, 2108.1821, 2115.5942, 2123.0063, 2130.4175, 2137.8289, 2145.239, 2152.6482, 2160.0576, 2167.467, 2174.8755, 2182.283, 2189.6904, 2197.097, 2204.5034, 2211.9092, 2219.3147, 2226.7195, 2234.1233, 2241.5269, 2248.9297, 2256.3328, 2263.7346, 2271.1365, 2278.5376, 2285.9387, 2293.3386, 2300.7378, 2308.136, 2315.5342, 2322.9326, 2330.3298, 2337.7263, 2345.1216, 2352.517, 2359.9126, 2367.3071, 2374.7007, 2382.0935, 2389.486, 2396.878, 2404.2695, 2411.6604, 2419.0513,
            #            2426.4402, 2433.8303, 2441.2183, 2448.6064, 2455.9944, 2463.3816, 2470.7678, 2478.153, 2485.5386, 2492.9238]
            wl_orig = ei.wavelengths
            # absorption region used for fitting the measurement to the model
            ind_fit = np.where((np.array(wl_orig) >= 2100) & (np.array(wl_orig) <= 2440))
            wl_only_in_fit = np.array(wl_orig)[ind_fit]

            # indices of wavelenghts outside of methane visibility
            ind_out = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224,
                       225, 226, 227, 228, 229, 230, 277, 278, 279, 280, 281, 282, 283, 284]
            mf = cmf.values

            plume_coord = None
            ite = 0 # if not 0 has special behaviour ... simulates plume shifts?
            ii = 0

            dist_opt = 1
            results = get_radiance_ratio(wl_orig, radius, num_pts, rdn, mf, plume_coord, orig_points_inside_plume, ind_out, ite, ii, combined_mask, background_mask, plume_mask, dist_opt)
            contour_coord, similarity_perpix, ratio, top_ind, top_mf, avg_top_mf, avg_in_plume_mf, top_pairs = results

            if debug_viz:
                # Fig 1: Plot selected points
                points_A = []
                points_B = []
                for pair in top_pairs:
                    # (x, y) of target pixel, (x ,y) of background pixel, similarity score
                    coord_a, coord_b, distance_similarity = pair
                    points_A.append([coord_a[1],coord_a[0]])
                    points_B.append([coord_b[1],coord_b[0]])
                points_A = np.asarray(points_A)
                points_B = np.asarray(points_B)

                mask_ = np.where(cmf.values == -9999, 1, 0)
                cmf_ = cmf.values / 8000.
                cmf_ = np.where(mask_ == 1, 0, cmf_)
                plt.imshow(cmf_, vmin=0, vmax=0.3)

                point_size = 12
                plt.scatter(points_A[:,0], points_A[:,1], s=point_size, color="orange", label="in-plume") # in plume
                plt.scatter(points_B[:,0], points_B[:,1], s=point_size, color="red", label="out-of-plume") # in background
                plt.legend()
                plt.show()

            ###
            sig = np.array(scene_target_signature)[ind_fit]
            ratio = np.array(ratio)[ind_fit]  # target-to-background radiance ratio
            coef, fit_sig = calculate_fit(deg_poly, wl_only_in_fit, sig, ratio)  # fit the model to measurement
            coef[0] *= 1e5  # multiply estimated concentration length with the scaling factor
            # coef[0] also known as alpha or as estim_concentration_len
            estim_concentration_len = coef[0]

            if debug_viz:
                # Fig 2: Show how well do we match spectrally
                fig, axes = plt.subplots(2, 1, constrained_layout=True, squeeze=False)
                axes = axes.flatten()

                wl = wl_only_in_fit
                yy = np.polyval(coef[1:], wl)  # continuum function
                y1 = ratio.copy()
                y2 = fit_sig.copy()
                ax_upper = axes[0]

                ax_upper.plot(wl, yy, label='Continuum function', color='green', linestyle='--', alpha=0.5)
                line1, = ax_upper.plot(wl, y1, label='Measurement')
                line2, = ax_upper.plot(wl, y2, label='Model')
                color1 = line1.get_color()
                color2 = line2.get_color()

                ax_lower = axes[1]
                y1 = y1 / yy
                y2 = y2 / yy
                ax_lower.plot(wl, y1, label='Measurement / continuum function', color=color1, linestyle='--')
                ax_lower.plot(wl, y2, label='Model / continuum function', color=color2, linestyle='--')
                ax_lower.set_xlabel('Wavelength')

                # legend:
                handles_upper, labels_upper = ax_upper.get_legend_handles_labels()
                handles_lower, labels_lower = ax_lower.get_legend_handles_labels()
                combined_handles = handles_upper + handles_lower
                combined_labels = labels_upper + labels_lower
                plt.legend(combined_handles, combined_labels,
                                fontsize=8)
                plt.show()

            polyn = np.polyval(coef[1:], wl_only_in_fit)
            ratio_p = ratio / polyn
            sig_p = fit_sig / polyn

            dist_t = calculate_dist(ratio_p, sig_p, 0)
            mag_t = calculate_magnitude(ratio_p, 1)
            dist_t /= mag_t

            print("D_norm = ", round(dist_t, 3)) # normalized distance between measurement and model ~ low => (more likely) plume
            print("Estimated_concentration_len = ", round(estim_concentration_len, 3)) # ~ similar to mf in that polygon, high => (more likely) plume

            # Somehow report the result...
            scores_per_polygon[pol_idx] = {"D_norm":dist_t,"alpha_con_len":estim_concentration_len}
    return scores_per_polygon