# -*- coding: utf-8 -*-
"""
Created on Tue Jul 02 17:52:48 2024

@author: Riaz Hussain, PhD

Master Script for analysis of 129Xenon ventilation MR images using
    1) Hierarchical k-means clustering as introduced in:
        Kirby et al., Academic Radiology, 2012
    2) Adaptive k-means clustering as introduced in:
        i) Zha et al., Academic Radiology, 2016
        ii) Zha et al., Academic Radiology, 2018
This script is a wrapper for the hierarchical.py and adaptive.py scripts 
"""

#%%Import Libraries and import single vdp functions file
import os
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from scipy.ndimage import gaussian_filter
import utils_funcs as ufunc

#%%Hierarchical Kmeans analysis
def hierarchical_kmeans(img_msk_prtn, kmeans_dir, analysis_corr_subjectid):
    """Master Function for hierarchical kmeans analysis"""
    mskd_img = img_msk_prtn[0] * img_msk_prtn[1]
    mskd_img /= np.max(mskd_img)
    analysis = "hierarchical-kmeans"
    _, _, corr_type, subject_id, _ = analysis_corr_subjectid
    if isinstance(kmeans_dir, list):
        dir_e_dtype = [kmeans_dir[0], corr_type]
    else:
        dir_e_dtype = [kmeans_dir, corr_type]
    all_clusters = calc_hkmeans_clusters(mskd_img, img_msk_prtn[1])
    hkmeans_hist_overlay(mskd_img, all_clusters, dir_e_dtype)
    kmeans_seg_arr, kmeans_prcnts = calc_hkmeans_bins(mskd_img, all_clusters,
                                                      data_type=corr_type)
    np.save(os.path.join(dir_e_dtype[0], f"{corr_type}_{analysis}_defect_array"),
            kmeans_seg_arr)
    kmeans_cmap_montage = ufunc.create_montage_array(kmeans_seg_arr, "all", img_msk_prtn[1])
    kmeans_fig = create_kmeans_montage(kmeans_cmap_montage, kmeans_prcnts, 5,
                                       img_msk_prtn[2])
    hkmean_heading = ["VDP", "LVP", "NVP1", "NVP2", "HVP"]
    # # #Save figure if directory provided
    if dir_e_dtype[0] is not None:
        kmeans_map = os.path.join(dir_e_dtype[0], f"{corr_type}_{analysis}_binning_map.png")
        print(f"Kmeans map image saved as: {kmeans_map}")
        kmeans_fig.savefig(kmeans_map, dpi=300, bbox_inches='tight', pad_inches=0)
    if dir_e_dtype[0] is not None:
        ufunc.save_txt_file(kmeans_prcnts, dir_e_dtype[0],
                            f'{corr_type}_{analysis}_percentages.txt',
                                hkmean_heading, subject_id)

def calc_hkmeans_clusters(img_msked, msk_in):
    """
    Function to perform hierarchical kmeans clustering based on
    Kirby et al., Academic Radiology, 2012
    """
    # Step 1: Initialize cluster centers and perform K-means clustering
    image_vector1 = img_msked[msk_in == 1]
    cluster_centers1 = (np.array([0.2, 0.4, 0.6, 0.8]) * (np.max(image_vector1)
                                - np.min(image_vector1)) + np.min(image_vector1))
    kmeans_round1 = KMeans(n_clusters=4, n_init=1, init=cluster_centers1.reshape(-1, 1))
    kmeans_round1.fit(image_vector1.reshape(-1, 1))
    # Step 2: Extract cluster 1 and perform K-means clustering again
    image_vector2 = image_vector1[kmeans_round1.labels_ == 0]
    cluster_centers2 = (np.array([0.2, 0.4, 0.6, 0.8]) * (np.max(image_vector2)
                                - np.min(image_vector2)) + np.min(image_vector2))
    kmeans_round2 = KMeans(n_clusters=4, n_init=1, init=cluster_centers2.reshape(-1, 1))
    kmeans_round2.fit(image_vector2.reshape(-1, 1))
    # Merge first two and last two clusters (make 2 clusters from 4)
    kmeans_round2.labels_[kmeans_round2.labels_ == 1] = 0
    kmeans_round2.labels_[kmeans_round2.labels_ == 2] = 1
    kmeans_round2.labels_[kmeans_round2.labels_ == 3] = 1
    five_clusters=[image_vector2[kmeans_round2.labels_==0],image_vector2[kmeans_round2.labels_==1],
                image_vector1[kmeans_round1.labels_==1], image_vector1[kmeans_round1.labels_==2],
                image_vector1[kmeans_round1.labels_==3]]
    return five_clusters

def hkmeans_hist_overlay(maskd_image, clusters, dir_e_tipi):
    """kmeans bin-data overlay"""
    data_path, data_type = dir_e_tipi
    img_1d = maskd_image[maskd_image != 0]
    mean_val = np.nanmean(img_1d)
    print(f"\nHierarchical - mean value: {mean_val}")
    _, ax = plt.subplots()
    ax.axvspan(np.min(maskd_image), np.max(clusters[0]), facecolor='#ff0000', alpha=1)
    ax.axvspan(np.max(clusters[0]),np.max(clusters[1]), facecolor='#ffb600', alpha=1)
    ax.axvspan(np.max(clusters[1]),np.max(clusters[2]), facecolor='#66b366', alpha=1)
    ax.axvspan(np.max(clusters[2]),np.max(clusters[3]), facecolor='#00ff00', alpha=1)
    ax.axvspan(np.max(clusters[3]),np.max(clusters[4]), facecolor='#0000ff', alpha=1)
    plt.hist(img_1d, bins=50, histtype='bar', density=True, color='white',
                edgecolor="black", alpha=0.6, label='Data')
    plt.axvline(mean_val, color='black', ls='-', lw=2, label='Mean')
    # plt.axvline(0.6*mean_val, color='black', ls='--', lw=2, label='Th60%')
    plt.legend()
    plt.xticks(fontsize=14, weight='bold')
    plt.yticks(fontsize=14, weight='bold')
    plt.xlabel('Normalized Intensity (arb.units)', fontsize=14, fontweight='bold')
    plt.ylabel('Fraction of Pixels', fontsize=14, fontweight='bold')
    plt.tight_layout()
    # Save the figure if directory is provided
    if data_path is not None:
        filename = os.path.join(data_path,
                    f"{data_type}_hierarchical_kmeans_hist_bins_overlay.png")
        print(f"Histogram saved as: {filename}")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def calc_hkmeans_bins(img_in, clstrs, data_type=None):
    """Calculate kmeans binning percents"""
    segmentation_img = np.zeros_like(img_in)
    segmentation_img[(img_in > 0) & (img_in < np.max(clstrs[0]))]= 1
    segmentation_img[(img_in >= np.max(clstrs[0])) &
                     (img_in < np.max(clstrs[1]))]= 2
    segmentation_img[(img_in >= np.max(clstrs[1])) &
                     (img_in < np.max(clstrs[2]))]= 3
    segmentation_img[(img_in >= np.max(clstrs[2])) &
                     (img_in < np.max(clstrs[3]))]= 4
    segmentation_img[img_in >= np.max(clstrs[3])]= 5
    total_count = np.sum(np.array(segmentation_img) > 0)
    vdp = np.round((((np.sum(np.array(segmentation_img)==1))/total_count)*100), 3)
    lvp = np.round((((np.sum(np.array(segmentation_img)==2))/total_count)*100), 3)
    nvp1 = np.round((((np.sum(np.array(segmentation_img)==3))/total_count)*100), 3)
    nvp2 = np.round((((np.sum(np.array(segmentation_img)==4))/total_count)*100), 3)
    hvp = np.round((((np.sum(np.array(segmentation_img)==5))/total_count)*100), 3)
    bins_prcnt = [vdp, lvp, nvp1, nvp2, hvp]
    bin_types = {"Defect": vdp, "Low": lvp, "Normal1": nvp1, "Normal2": nvp2,
                "Hyper": hvp}
    # Print the results using a loop
    for vent_type, prcntage in bin_types.items():
        print(f"{data_type} hierarchical-kmeans {vent_type} (%): {prcntage}\n")
    return segmentation_img, bins_prcnt

#%%Adaptive Kmeans analysis
def adaptive_kmeans(img_msk_prtn, kmeans_dir, analysis_corr_subjectid):
    """Master Function for adaptive kmeans analysis"""
    mskd_img = img_msk_prtn[0] * img_msk_prtn[1]
    mskd_img /= np.max(mskd_img)
    analysis = "adaptive-kmeans"
    _, _, corr_type, subject_id, _ = analysis_corr_subjectid
    if isinstance(kmeans_dir, list):
        dir_e_dtype = [kmeans_dir[1], corr_type]
    else:
        dir_e_dtype = [kmeans_dir, corr_type]
    all_clusters = calc_akmeans_clusters(mskd_img, img_msk_prtn[1])
    akmeans_hist_overlay(mskd_img, all_clusters, dir_e_dtype)
    kmeans_seg_arr, kmeans_prcnts = calc_akmeans_bins(mskd_img, all_clusters,
                                                      data_type=corr_type)
    np.save(os.path.join(dir_e_dtype[0], f"{corr_type}_{analysis}_defect_array"),
            kmeans_seg_arr)
    kmeans_cmap_montage = ufunc.create_montage_array(kmeans_seg_arr, "all", img_msk_prtn[1])
    akmeans_fig = create_kmeans_montage(kmeans_cmap_montage, kmeans_prcnts, 4, img_msk_prtn[2])
    vent_kmean_heading = ["VDR", "LVR", "MVR", "HVR"]
    # # #Save figure if directory provided
    if dir_e_dtype[0] is not None:
        kmeans_map = os.path.join(dir_e_dtype[0], f"{corr_type}_{analysis}_binning_map.png")
        print(f"Kmeans map image saved as: {kmeans_map}")
        akmeans_fig.savefig(kmeans_map, dpi=300, bbox_inches='tight', pad_inches=0)
        ufunc.save_txt_file(kmeans_prcnts, dir_e_dtype[0],
                            f'{corr_type}_{analysis}_percentages.txt',
                            vent_kmean_heading, subject_id)

def calc_akmeans_clusters(mr_img: np.ndarray, bin_mask: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform adaptive KMeans clustering based on Zha et al. 2016 & 2018
    Parameters:
    masked_img (np.ndarray): The masked image.
    bin_mask (np.ndarray): The binary mask.
    Returns:
    The vent defect voxels, low, medium, and high vent voxels.
    """
    img_1d = mr_img[bin_mask == 1]
    lowp, cluster_1 = calculate_pl(img_1d)
    vdr = calculate_vdr(img_1d, lowp)
    ##Next k-means clustering applied to ventilated volume
    vent_vector = img_1d[img_1d > np.max(cluster_1)]
    cluster_centers3 = (np.array([0.167, 0.333, 0.5, 0.667, 0.833]) *
                        (np.max(vent_vector) - np.min(vent_vector))
                        + np.min(vent_vector))
    kmeans_round3 = fit_kmeans(vent_vector, cluster_centers3)
    lvr, mvr, hvr = classify_vent_regions(vent_vector, kmeans_round3, lowp)
    clstrs = [vdr, lvr, mvr, hvr]
    return clstrs

def fit_kmeans(im_vec: np.ndarray, cluster_centers: np.ndarray) -> KMeans:
    """
    Perform KMeans clustering with the given initial cluster centers.
    Parameters:
    im_vec (np.ndarray): The input image vector.
    cluster_centers (np.ndarray): The initial cluster centers.
    Returns:
    KMeans: The fitted KMeans model.
    """
    kmeans = KMeans(n_clusters=len(cluster_centers), n_init=1,
                        init=cluster_centers.reshape(-1, 1))
    kmeans.fit(im_vec.reshape(-1, 1))
    return kmeans

def calculate_pl(img_vector: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Determine P_L value from 1D image array to be used for adaptive kmeans.
    Parameters:
    img_vector (np.ndarray): The input 1D image vector.
    Returns:
    Tuple[float, np.ndarray]: The P_L value and the first cluster.
    """
    hist, bin_edges = np.histogram(img_vector, bins=10)
    bin_1 = img_vector[img_vector < bin_edges[1]]
    p_l = (hist[0] / sum(hist)) * 100
    print(f"P_L (%) = {p_l}")
    plt.hist(img_vector, bins=10, histtype='bar', density=True, color='black',
            edgecolor="cyan")
    plt.xticks(fontsize=14, weight='bold')
    plt.yticks(fontsize=14, weight='bold')
    plt.xlabel('Normalized Intensity (arb.units)', fontsize=14, fontweight='bold')
    plt.ylabel('Fraction of Pixels', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return p_l, bin_1

def calculate_vdr(img_vector: np.ndarray, p_l: int) -> np.ndarray:
    """
    Calculate VDR based on the given case.
    Parameters:
    img_vector (np.ndarray): The input image vector.
    case (int): The case number (1, 2, or 3).
    Returns:
    np.ndarray: The vent defect region.
    """
    if p_l < 4:
        print("P_L is less than 4%.. Round 1: 5 clusters")
        cluster_centers1 = (np.array([0.167, 0.333, 0.5, 0.667, 0.833]) *
                            (np.max(img_vector) - np.min(img_vector))
                            + np.min(img_vector))
    else:
        print("P_L is greater than 4%... Round 1: 4 clusters")
        cluster_centers1 = (np.array([0.2, 0.4, 0.6, 0.8]) *
                            (np.max(img_vector) - np.min(img_vector))
                            + np.min(img_vector))
    ##Run keamns round 1
    kmeans_round1 = fit_kmeans(img_vector, cluster_centers1)
    img_vector2 = img_vector[kmeans_round1.labels_ == 0]
    ##Case: P_L < 10, run second round of kmeans
    if p_l < 10:
        cluster_centers2 = (np.array([0.2, 0.4, 0.6, 0.8]) *
                            (np.max(img_vector2) - np.min(img_vector2)
                             ) + np.min(img_vector2))
        kmeans_round2 = fit_kmeans(img_vector2, cluster_centers2)
    ##Depending on P_L, re-label clusters
        if p_l < 4:
            print("P_L is less than 4%... Round 2 re-labeling")
            kmeans_round2.labels_[kmeans_round2.labels_ == 1] = 0
            kmeans_round2.labels_[kmeans_round2.labels_ == 2] = 1
            kmeans_round2.labels_[kmeans_round2.labels_ == 3] = 1
        else:
            print("P_L is between 4 - 10%... Round 2 re-labeling")
            kmeans_round2.labels_[kmeans_round2.labels_ == 1] = 0
            kmeans_round2.labels_[kmeans_round2.labels_ == 2] = 0
            kmeans_round2.labels_[kmeans_round2.labels_ == 3] = 1
        vent_defect_region = img_vector2[kmeans_round2.labels_ == 0]
    else:
        print("P_L is greater than 10%... No Round 2 for VDR")
        vent_defect_region = img_vector[kmeans_round1.labels_ == 0]
    return vent_defect_region

def classify_vent_regions(vent_vector: np.ndarray, kmeans_labels: KMeans,
                          lowp: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Classify vent regions based on P_L value.
    Parameters:
    vent_vector (np.ndarray): The input vent vector.
    kmeans_labels (KMeans): The KMeans labels from previous clustering.
    lowp (float): The P_L value.
    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray]: Low, medium, and high ventilation regions.
    """
    if lowp < 10:
        print("Round 3: P_L < 10%. Bigger low_vent_region")
        img_vector = vent_vector[kmeans_labels.labels_ == 1]
    else:
        print("Round 3: P_L > 10%. Bigger high_vent_region")
        img_vector = vent_vector[kmeans_labels.labels_ == 3]
    ##Run kmeans on ventilated voxels only
    cluster_centers = (np.array([0.167, 0.333, 0.5, 0.667, 0.833]) *
                       (np.max(img_vector) - np.min(img_vector))
                       + np.min(img_vector))
    kmeans_round4 = fit_kmeans(img_vector, cluster_centers)
    ##Re-label clusters and gather LVR, MVR, and HVR voxels
    if lowp < 10:
        kmeans_round4.labels_[kmeans_round4.labels_ == 1] = 0
        kmeans_round4.labels_[kmeans_round4.labels_ == 2] = 1
        kmeans_round4.labels_[kmeans_round4.labels_ == 3] = 1
        kmeans_round4.labels_[kmeans_round4.labels_ == 4] = 1
        low_vent_region = np.concatenate((vent_vector[(kmeans_labels.labels_ == 0)],
                                          img_vector[(kmeans_round4.labels_ == 0)]))
        medium_vent_region = np.concatenate((vent_vector[(kmeans_labels.labels_ == 2)],
                                             img_vector[(kmeans_round4.labels_ == 1)]))
        high_vent_region = vent_vector[(kmeans_labels.labels_ == 3) | (kmeans_labels.labels_ == 4)]
    else:
        kmeans_round4.labels_[kmeans_round4.labels_ == 1] = 0
        kmeans_round4.labels_[kmeans_round4.labels_ == 2] = 0
        kmeans_round4.labels_[kmeans_round4.labels_ == 3] = 1
        kmeans_round4.labels_[kmeans_round4.labels_ == 4] = 1
        low_vent_region = vent_vector[(kmeans_labels.labels_ == 0) | (kmeans_labels.labels_ == 1)]
        medium_vent_region = np.concatenate((vent_vector[(kmeans_labels.labels_ == 2)],
                                             img_vector[(kmeans_round4.labels_ == 0)]))
        high_vent_region = np.concatenate((vent_vector[(kmeans_labels.labels_ == 4)],
                                           img_vector[(kmeans_round4.labels_ == 1)]))
    return low_vent_region, medium_vent_region, high_vent_region

def akmeans_hist_overlay(maskd_image, clusters, dir_e_tipi):
    """kmeans bin-data overlay"""
    data_path, data_type = dir_e_tipi
    img_1d = maskd_image[maskd_image != 0]
    mean_val = np.nanmean(img_1d)
    print(f"\nAdaptive - mean value: {mean_val}")
    _, ax = plt.subplots()
    ax.axvspan(np.min(maskd_image), np.max(clusters[0]), facecolor='#ff0000', alpha=1)
    ax.axvspan(np.max(clusters[0]),np.max(clusters[1]), facecolor='#ffb600', alpha=1)
    ax.axvspan(np.max(clusters[1]),np.max(clusters[2]), facecolor='#66b366', alpha=1)
    ax.axvspan(np.max(clusters[2]),np.max(clusters[3]), facecolor='#0000ff', alpha=1)
    plt.hist(img_1d, bins=50, histtype='bar', density=True, color='white',
                edgecolor="black", alpha=0.6, label='Data')
    plt.axvline(mean_val, color='black', ls='-', lw=2, label='Mean')
    plt.axvline(0.6*mean_val, color='black', ls='--', lw=2, label='Th60%')
    plt.legend()
    plt.xticks(fontsize=14, weight='bold')
    plt.yticks(fontsize=14, weight='bold')
    plt.xlabel('Normalized Intensity (arb.units)', fontsize=14, fontweight='bold')
    plt.ylabel('Fraction of Pixels', fontsize=14, fontweight='bold')
    plt.tight_layout()
    # Save the figure if directory is provided
    if data_path is not None:
        filename = os.path.join(data_path,
                    f"{data_type}_adaptive_kmeans_hist_bins_overlay.png")
        print(f"Histogram saved as: {filename}")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def calc_akmeans_bins(img_in, clstrs, data_type=None):
    """Calculate kmeans binning percents"""
    img_in /= np.max(img_in)
    segmentation_img = np.zeros_like(img_in)
    segmentation_img[(img_in > 0) & (img_in < np.max(clstrs[0]))] = 1
    segmentation_img[(img_in >= np.max(clstrs[0])) &
                     (img_in < np.max(clstrs[1]))] = 2
    segmentation_img[(img_in >= np.max(clstrs[1])) &
                     (img_in < np.max(clstrs[2]))] = 3
    segmentation_img[img_in >= np.max(clstrs[2])] = 4
    total_count = np.sum(np.array(segmentation_img) > 0)
    vdr = np.round((((np.sum(np.array(segmentation_img)==1))/total_count)*100), 3)
    lvr = np.round((((np.sum(np.array(segmentation_img)==2))/total_count)*100), 3)
    mvr = np.round((((np.sum(np.array(segmentation_img)==3))/total_count)*100), 3)
    hvr = np.round((((np.sum(np.array(segmentation_img)==4))/total_count)*100), 3)
    bins_prcnt = [vdr, lvr, mvr, hvr]
    bin_types = {"VDR": vdr, "LVR": lvr, "MVR": mvr, "HVR": hvr}
    # Print the results using a loop
    for vent_type, prcntage in bin_types.items():
        print(f"{data_type} adaptive-kmeans {vent_type} (%): {prcntage}")
    return segmentation_img, bins_prcnt

#%%Common functions
def create_kmeans_montage(kmeans_cmap_montage, bin_percents, colrs=5, prtn_arr=None):
    """Montage for kmeans binning maps"""
    # Define color map with 0 values set to black and other values assigned to colors
    kmean_cm = ufunc.binning_cmap(bin_percents, colrs, prtn_arr)
    akmean_fig, ax = plt.subplots(figsize=(16, 3), dpi=300)
    ##protom data is already in montage format (corresponding to non-zero mask slices)
    if prtn_arr is not None and np.any(prtn_arr):
        blurred_proton = gaussian_filter(prtn_arr, sigma=6)
        ax.imshow(blurred_proton, cmap='gray', alpha=1)
    ax.imshow(kmeans_cmap_montage, cmap=kmean_cm, alpha=1)
    plt.axis('off')
    sm = ufunc.fixed_cbar(colrs)
    # # Create an axis for the colorbar using make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1.5%", pad="0.1%")
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_ticklabels([])
    cbar.ax.tick_params(size=0)
    plt.show(block=False)
    return akmean_fig
