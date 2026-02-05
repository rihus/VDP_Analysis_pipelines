# -*- coding: utf-8 -*-
"""
Created on Wed Jul 03 11:52:48 2024

@author: Riaz Hussain, PhD

Script for analysis of 129Xenon ventilation MR images Using
    (1) Percent of whole-lung-mean thresholding as introduced in:
        Thomen et al., Journal of Cystic Fibrosis, 2017

"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage import io, img_as_ubyte
import utils_funcs as ufunc

#%%The 60% of mean thresholding analysis
def mean_thresholding(threshold_dir, norm_img, msk_arr, threshold_pair,
                           analyses_corr_subjectid=None):
    """Main script to run all thresholding vdp functions"""
    print("Running mean--thresholded vdp analysis")
    vdp_threshold, hvp_threshold = threshold_pair
    _, _, corr_type, subject_id, _ = analyses_corr_subjectid
    ##Gerenrate defects array based on provided thresholds of mean(image)
    threshold_arr = create_threshold_array(norm_img, msk_arr, vdp_threshold, hvp_threshold)
    # Save thresholding defect array data as numpy file for later use
    np.save(os.path.join(threshold_dir, f"{corr_type}_thresholding_defect_array"),
            threshold_arr)
    # #Calculate vdp/hvp numbers from the array
    vdp = calc_vdp(threshold_arr, msk_arr)
    hvp = calc_hvp(threshold_arr, msk_arr)
    # #Save text file with vdp, hvp results
    save_vdp_hvp(vdp, hvp, threshold_dir, corr_type, subject_id)
    # #Overlay defect/hyper pixels on top of the mr image and make montage array
    defect_hyper_overlay_array = overlay_images(norm_img, threshold_arr)
    defect_hyper_overlay_montage = ufunc.create_montage_array(defect_hyper_overlay_array,
                                                              msk_file=msk_arr)
    # #Plot the overlaid montage image and save it in the appropriate directory
    vdphvp_4d=ufunc.montage_plot_4d(defect_hyper_overlay_montage)
    ufunc.save_image(vdphvp_4d, threshold_dir, corr_type, im_type='vdp_hvp_overlay')
    save_slicewise_imgs(defect_hyper_overlay_array, threshold_dir, corr_type, im_type='overlay')

def create_threshold_array(mr_img, msk_array, vdp_threshold=None, hvp_threshold=None,
                           median_filter=True):
    """
    Calculate the vdp/hvp from 129Xe ventilation image array.
    Parameters:
    mr_img (ndarray): mri image of the lung
    msk_array (ndarray): binary mask of the lung region
    vdp_threshold/hvp_threshold (floats): numbers specifying thresholds
    median_filter: if median filter is to be applied (default: True)
    Returns:
    calculated defect array
    """
    # Ensure only useful signal is operated on:
    maskd_image = mr_img * msk_array
    # Calculate mean of non-zero masked image
    img_1d = maskd_image[maskd_image != 0]
    mean_val = np.nanmean(img_1d)
    print(f"\nThresholding - mean value: {mean_val}")
    # Validate and fit thresholds within 0-1
    # #VDP Loop
    if vdp_threshold is not None:
        vent_threshold= vdp_threshold/100 if vdp_threshold>1 else vdp_threshold[1]
        if median_filter:
            vdp_arr = (maskd_image < (mean_val * vent_threshold) * (msk_array>0)).astype(int)
            # Calculate the defect ventilation percentage with median filter
            vent_defect = ufunc.med_filter(vdp_arr, nome="defect_array")
        else:
            # Calculate ventilation defct percentage without median filter
            vent_defect = (maskd_image < (np.nanmean(maskd_image[maskd_image>0])
                                            * vent_threshold) * (msk_array > 0)).astype(int)
    else:
        vent_defect = np.zeros_like(msk_array)
    ## HVP Loop
    if hvp_threshold:
        hyper_threshold= hvp_threshold/100 if hvp_threshold>1 else hvp_threshold[2]
        if median_filter:
            #Count hyper ventilation pixels depending on if median filter is applied or not.
            hvp_arr = (maskd_image > (mean_val * hyper_threshold) * (msk_array>0)).astype(int)
            #Calculate the hyper ventilation percentage with median filter
            vent_hyper = ufunc.med_filter(hvp_arr, nome="hyper_array")
        else:
            vent_hyper = (maskd_image > (np.nanmean(maskd_image[maskd_image>0])
                                        * hyper_threshold) * (msk_array > 0)).astype(int)
    else:
        vent_hyper = np.zeros_like(msk_array)
    # get the defect-hyper array
    defect_hyper_arr = vent_defect + (4 * vent_hyper)
    return defect_hyper_arr

def calc_vdp(defect_array, msk_array):
    """
    Count the Ventilation Defect Percentage from defect array.
    Parameters:
    defect array (ndarray): defect array from create_defect_array function.
    mask_array (ndarray): binary mask of the lung region.
    Returns:
    vdp (float): calculated vdp
    """
    total_count = np.sum(np.array(msk_array) > 0)
    vdp = np.round((((np.sum(np.array(defect_array)==1))/total_count)*100), 3)
    return vdp

def calc_hvp(defect_array, msk_array):
    """
    Count the Ventilation Hyperventilated Percentage from defect array.
    Parameters:
    defect array (ndarray): defect array from create_defect_array function.
    mask_array (ndarray): binary mask of the lung region.
    Returns:
    hyper (float): calculated hvp
    """
    total_count = np.sum(np.array(msk_array) > 0)
    hvp = np.round((((np.sum(np.array(defect_array)==4))/total_count)*100), 3)
    return hvp

def save_vdp_hvp(vdp, hvp, path_in, data_type=None, subject_id=None):
    """arrange and save text file of vdp/hvp results"""
    normal = 100 - (vdp + hvp)
    vdp_results_all = [vdp, hvp, normal]
    vdp_titles = ["Subject_id",'VDP', 'HVP', 'NVP']
    print(f"\n{subject_id} {data_type}\n \t VDP: {vdp}%,\n \t HVPP: {hvp}%,"
          f"\n \t Normal: {normal}%")
    #Write ventilation defect percentage (VDP) and hyperventilated percentage (HVP) to files
    all_vdp_name = ("vdp_thresh_results.txt" if data_type is None
                    else f"{data_type}_thresholding_results.txt")
    ufunc.save_txt_file(vdp_results_all, path_in, all_vdp_name,
                        vdp_titles, subject_id)

def overlay_images(norm_mr_image, defect_array):
    """
    Overlay 3D RGB image array on each 2D slice of 3D MRI image array and save as PNG image file.

    Args:
        mr_image (numpy.ndarray): normalized MRI image array of shape (height, width, depth).
        defect_array (np.ndarray): Array of shape (height, width, depth) with integer values
        representing different defect types (0 normal, 1 incomplete, 2 complete, and 4 hyperintense)
        mask_image (np.ndarray): Binary mask shape(height, width, depth) for normalizing the image.
        output_dir (str): Directory to save the output PNG image files.
        data_type (str): type of MR image data; Options: None (default), Non_corr, N4_corr, FA_corr
    Returns:
        4D array of Non_corr MR image overlayed by defects
    """
    defect_types = {0: 'Normal', 1: 'Defect', 4: 'Hyper'}
    defect_dict = {defect_type: (defect_array == index).astype(int)
                    for index, defect_type in defect_types.items()}
    ##Define a color map
    cmap_list = [[[0, 0, 0], [1, 0, 0]], [[0, 0, 0], [0, 0, 1]], [[0, 0, 0], [1, 1, 1]]]
    cm_defect, cm_hyper, cm_normal = [ListedColormap(cm) for cm in cmap_list]
    # This part uses transpose to change the dimensions of each array,
    # applies color map function (cm_defect, cm_hyper, and cm_normal) to each.
    # Finally, the result is transposed again to obtain the desired shape of the output array.
    array_defect=cm_defect(defect_dict['Defect'].transpose(2,0,
                                                            1))[..., :3].transpose(1,2,0,3)
    array_hyper=cm_hyper(defect_dict['Hyper'].transpose(2,0,1))[..., :3].transpose(1,2,0,3)
    array_normal=cm_normal(defect_dict['Normal'].transpose(2,0,1))[..., :3].transpose(1,2,0,3)
    array4d_rgb = np.stack([norm_mr_image] * 3, axis=-1)
    defect_array_rgb = np.sum([array_defect, 4 * array_hyper], axis=0)
    defect_overlay_4d = ((defect_array_rgb + (array_normal * array4d_rgb)) * 255).astype(np.uint8)
    return defect_overlay_4d

def save_slicewise_imgs(image_in, output_dir, data_type=None, im_type=None):
    """Save slice-wise images of of a 3d/4d montage"""
    for an_img in range(image_in.shape[2]):
        out_file = (f"img_slice_{an_img}.png" if data_type is None
                    else f"{data_type}_{im_type}_img_slice_{an_img}.png")
        full_file_name = os.path.join(output_dir, out_file)
        if image_in.ndim == 3:
            current_slice = image_in[:,:,an_img]
            # Save the image using Matplotlib
            plt.imshow(current_slice, cmap='gray')
            plt.axis('off')
            plt.savefig(full_file_name, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()
        if image_in.ndim == 4:
            io.imsave(full_file_name, img_as_ubyte(image_in[:,:,an_img,:]), check_contrast=False)

#%%
