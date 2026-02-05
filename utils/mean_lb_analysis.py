# -*- coding: utf-8 -*-
"""
Created on Tue Jul 02 17:52:48 2024

@author: Riaz Hussain, PhD

Master Script for analysis of 129Xenon ventilation MR images using
    1) 99th percentile normalized Linear binning as introduced in:
        (1) He et al., Academic Radiology, 2014
        (2) He et al., Academic Radiology, 2016
    2) Mean normalized Linear binning as introduced in:
        (1) Collier et al., ISMRM Abstract#4482, 2018
        (2) Collier et al., ISMRM Abstract#0442, 2020

"""

#%%Import Libraries and import single vdp functions file
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import utils_funcs as ufunc

#%%Mean normalized linear binning analysis
def mean_lb(img_msk_prtn, glb_dirs, analysis_corr_subjectid):
    """Master function for LB mean analysis"""
    mskd_img = img_msk_prtn[0] * img_msk_prtn[1]
    mskd_img /= np.max(mskd_img)
    mskd_norm_img_prtn = [mskd_img, img_msk_prtn[2]]
    if isinstance(glb_dirs, list):
        dirs_n_analysis = [glb_dirs[1], 'lb-mean']
    else:
        dirs_n_analysis = [glb_dirs, 'lb-mean']
    # # ##Calculate linear binning VDP/HVP - mean
    main_lb_mean_func(mskd_norm_img_prtn, dirs_n_analysis, analysis_corr_subjectid)

def main_lb_mean_func(msked_image_prtn, dirs_e_analysis, analyses_correc_subjectid):
    """Master function for mean glb"""
    msked_image, prtn = msked_image_prtn
    img_1d = msked_image[msked_image != 0]
    thresholds = [0.33, 0.66, 1.0, 1.33, 1.66]
    # #Plot the single subject histogram with mean normalized overlay
    mean_val, mean_prcntld_val, mean_1d_arr = mean_histograms(img_1d, dirs_e_analysis[0],
                                                                    analyses_correc_subjectid[2])
    # # #Make any values above 99th percentile of mean equal to max in the image
    image_mean_nrmlzd = np.array(msked_image / mean_val)
    image_mean_nrmlzd[image_mean_nrmlzd > mean_prcntld_val] = mean_prcntld_val
    img_mean_e_prtn = [image_mean_nrmlzd, prtn]
    # ##Overlaying healthy-cohort-fit on single subjects' histogram (mean normalized)
    hist_with_bins(mean_1d_arr, thresholds, dirs_e_analysis,
                   analyses_correc_subjectid)
    binning_cmap_image(img_mean_e_prtn, thresholds, dirs_e_analysis,
                       analyses_correc_subjectid)

def mean_histograms(img_1d_array, mean_dir=None, data_type=None, prcntl_val=99):
    """
    Plots histograms from input 1D array: 1) all data 2) post mean normlization.
    The two (mean then percentile-cut) histograms are overlaid on top of each other.
    Parameters:
    img_1d_array: A 1D NumPy array of data.
    mean_dir: (Optional) Directory to save histograms.
    data_type: (Optional) String to specifies data type in array and to name output files.
    prcntl_val = (Optional) percentile value post mean normalization (Default=99th).
    Returns:
    mean_value: The mean value of the input array.
    oned_mean_prcntl_arr: The mean normalized, 99th percentile 1d array.
    """
    # Get the mean value of 1d image
    mean_val = np.mean(img_1d_array)
    #Normalize with mean
    mean_norm_arr = img_1d_array / mean_val
    plt.hist(mean_norm_arr, bins=100, histtype='bar', density=True, color='y')
    #get 99th percentile of the mean normalized array
    mean_prcntld_val = np.percentile(mean_norm_arr, prcntl_val, axis = 0)
    #Make anything above 99th percentile = 99th percentile (post mean normalization)
    mean_norm_arr[mean_norm_arr >= mean_prcntld_val] = mean_prcntld_val
    plt.hist(mean_norm_arr, bins=100, histtype='bar', density=True, color='c')
    # Save overlaid histograms if path provided
    if mean_dir is not None:
        filename = os.path.join(mean_dir, f"{data_type}_mean_hist_overlay.png")
        print(f"Histogram saved as: {filename}")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    return mean_val, mean_prcntld_val, mean_norm_arr

#%%Common functions
def hist_with_bins(image_1d, thresholds, dirs_e_analysis=None,
                        analyses_correc_subjectid=None):
    """
    Plots bins, histogram of a 1D array, and a curve on top, and saves the figure.
    Parameters:
        image_1D: 1D array to plot histogram and curve on top of
        thresholds: Dictionary of threshold values to create bins, keys are data types
        fit_params: Dictionary of fit parameters for plot (mean, sd, skewness), keys are data types
        data_path: str, optional (default=None)
            Directory name to save the figure. If not provided, figure will not be saved.
        dirs_e_analysis: str, optional (default=None)
            Data/Analysis type for creating the filename of saved figure
        correc_subjectid
    """
    data_path, analysis = dirs_e_analysis
    _, _, data_type, _, _ = analyses_correc_subjectid
    print(f"\nMaking {analysis}--normalized histogram-bins overlay")
    # Create colored threshold bins
    _, ax = plt.subplots()
    ax.axvspan(min(image_1d), thresholds[0], facecolor='#ff0000', alpha=1)
    ax.axvspan(thresholds[0],thresholds[1], facecolor='#ffb600', alpha=1)
    ax.axvspan(thresholds[1],thresholds[2], facecolor='#66b366', alpha=1)
    ax.axvspan(thresholds[2],thresholds[3], facecolor='#00ff00', alpha=1)
    ax.axvspan(thresholds[3],thresholds[4], facecolor='#0091b5', alpha=1)
    if thresholds[4] < max(image_1d):
        ax.axvspan(thresholds[4], max(image_1d), facecolor='#0000ff', alpha=1)
    mean_val = np.nanmean(image_1d)
    print(f"\n {analysis} - mean value: {mean_val}")
    # Plot the histogram
    plt.hist(image_1d, bins=50, histtype='bar', density=True, color='white',
             edgecolor="black", alpha=0.6, label='Data')
    plt.legend()
    plt.xticks(fontsize=14, weight='bold')
    plt.yticks(fontsize=14, weight='bold')
    plt.xlabel('Normalized Intensity (arb.units)', fontsize=14, fontweight='bold')
    plt.ylabel('Fraction of Pixels', fontsize=14, fontweight='bold')
    plt.tight_layout()
    # Save the figure if directory is provided
    if data_path is not None:
        filename = os.path.join(data_path, f"{data_type}_{analysis}_hist_bins_overlay.png")
        print(f"Histogram saved as: {filename}")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def binning_cmap_image(norm_prtn, thresholds, dirs_analysis=None,
                       analyses_correc_subjectid=None):
    """
    Create a new 3D image/array (color map) based on the input thresholds.
    Args:
        image_3d: 3D array
        thresholds: dictionary with threshold values for creating color bins, keys = data types.
        data_path: directory path to save output files. Defaults to None.
        prtn_arr: proton image to overlay with colormap. Default none.
        data_type: string to include in output filenames. Defaults to None.
        corr_subjectid_prtn: correction, subject name and proton file.
        analysis: string to include analysis type in output filenames. Defaults to None.
    """
    data_path, analysis = dirs_analysis
    _, _, corr_type, subject_id, _ = analyses_correc_subjectid
    binned_cmap_img=create_binned_img(norm_prtn[0], thresholds)
    # Save defect array data as numpy file for later use
    np.save(os.path.join(data_path, f"{corr_type}_{analysis}_defect_array"),
            binned_cmap_img)
    binning_percents = bin_counts(binned_cmap_img, corr_type)
    vent_binning_heading = ["VDP","LVP", "NVP1","NVP2", "EVP","HVP"]
    binned_cmap_montage = ufunc.create_montage_array(binned_cmap_img, "all", binned_cmap_img)
    binning_fig = create_binning_montage(binned_cmap_montage, binning_percents, norm_prtn[1])
    # # #Save figure if directory provided
    if data_path is not None:
        binning_map = os.path.join(data_path, f"{corr_type}_{analysis}_binning_map.png")
        print(f"Binning map image saved as: {binning_map}")
        binning_fig.savefig(binning_map, dpi=300, bbox_inches='tight', pad_inches=0)
    if data_path is not None:
        ufunc.save_txt_file(binning_percents, data_path, f'{corr_type}_{analysis}_percentages.txt',
                  vent_binning_heading, subject_id)

def create_binned_img(norm_mr_image, thresholds):
    """Create binned array from provided binning thresholds"""
    binned_img = np.zeros_like(norm_mr_image)
    #Fill the colormap according to thresholds
    binned_img[(norm_mr_image > 0) & (norm_mr_image < thresholds[0])] = 1
    binned_img[(norm_mr_image >= thresholds[0]) &
                (norm_mr_image < thresholds[1])] = 2
    binned_img[(norm_mr_image >= thresholds[1]) &
                (norm_mr_image < thresholds[2])] = 3
    #notice change of direction for the equal sign
    binned_img[(norm_mr_image >= thresholds[2]) &
                      (norm_mr_image <= thresholds[3])] = 4
    binned_img[(norm_mr_image > thresholds[3]) &
                      (norm_mr_image <= thresholds[4])] = 5
    binned_img[(norm_mr_image > thresholds[-1])]= len(thresholds)+1
    return binned_img

def bin_counts(binned_cmap_img, data_type):
    """
    Count and print various bin counts
    """
    # Calculate pixel counts and percentages
    nonzero_pixels = np.sum(np.array(binned_cmap_img) > 0)
    defect_prcnt = np.round((((np.sum(np.array(binned_cmap_img)==1))/nonzero_pixels)*100), 3)
    low_prcnt = np.round(((np.sum(np.array(binned_cmap_img)==2))/nonzero_pixels)*100, 3)
    normal1_prcnt = np.round(((np.sum(np.array(binned_cmap_img)==3))/nonzero_pixels)*100, 3)
    normal2_prcnt = np.round(((np.sum(np.array(binned_cmap_img)==4))/nonzero_pixels)*100, 3)
    high_prcnt = np.round(((np.sum(np.array(binned_cmap_img)==5))/nonzero_pixels)*100, 3)
    hyper_prcnt = np.round(((np.sum(np.array(binned_cmap_img)==6))/nonzero_pixels)*100, 3)
    bin_types = {"Defect": defect_prcnt, "Low": low_prcnt,
    "Normal 1": normal1_prcnt, "Normal 2": normal2_prcnt,
    "High": high_prcnt,"Hyper": hyper_prcnt}
    # Print the results using a loop
    for vent_type, prcntage in bin_types.items():
        print(f"{data_type} {vent_type} (%): {prcntage}")
    binning_prcnt_data = [defect_prcnt, low_prcnt, normal1_prcnt,
                          normal2_prcnt, high_prcnt, hyper_prcnt]
    return binning_prcnt_data

def create_binning_montage(binned_cmap_montage, bin_prcnts, prtn_arr=None):
    """Make montage for linear binning maps"""
    # Define color map with 0 values set to black and other values assigned to colors
    bins_cmap = ufunc.binning_cmap(bin_prcnts, 6, prtn_arr)
    # Create the montage with a fixed colorbar i.e. colors in cbar are fixed,
    # regardless of the fact that they're present in the plot or not.
    bin_fig, ax = plt.subplots(figsize=(16, 3), dpi=300)
    ##protom data is already in montage format (corresponding to non-zero mask slices)
    if prtn_arr is not None and np.any(prtn_arr):
        # binned_proton_montage = ufunc.create_montage_array(prtn_arr, 'all')
        blurred_proton = gaussian_filter(prtn_arr, sigma=6)
        ax.imshow(blurred_proton, cmap='gray', alpha=1)
    ax.imshow(binned_cmap_montage, cmap=bins_cmap, alpha=1)
    plt.axis('off')
    sm = ufunc.fixed_cbar(6)
    # # Create an axis for the colorbar using make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1.5%", pad="0.1%")
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_ticklabels([])
    cbar.ax.tick_params(size=0)
    plt.show(block=False)
    return bin_fig

#%%
