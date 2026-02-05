#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 7:45:48 2023

@author: Riaz Hussain, PhD

This script allows four types of analysis of the 129Xe ventilation MR images:
    1. The 60% of the total image intesity mean thresholding
    2. Generalized linear binning - 99th Percentile normalized
    3. Generalized linear binning - Mean normalized
    4. Hierarchical K-means clustering
    5. Adaptive K-means clustering
    6. Linear binning - mean normalized (fixed thresholds)
    
    Last updated: Feb 5 2026

"""
#%%Import Libraries and import master vdp_hvp functions file
import os
import re
#Import file with all the vdp function
from utils.master_vdp_functions_file import single_subj_analysis
from utils.utils_funcs import append_save_txt2csv, append_save_snr_txts

#%%Decalre analysis mode, subject directory names, MR Sequence type,
## Bias correction type, types of VDP/HVP analysis, SNR/MSR switch,
## Main directory, File name patterns, and thresholds
ANALYSIS_MODE = "Single" # Options: "Single", "Batch"
MR_SEQUENCE = "Spiral" # Options: "Cartesian", "Spiral"
##Provide directory name/s with subject data (pattern or full names list)
##Batch processing only
SUBJ_DIR_NAMES = ""
##Select bias field correction
CORR_TYPE = "corrected" # Options: "Non_corr", "N4_corr", "FA_corr" (only for Spiral), "corrected", "all"
##VDP analysis correction
VDP_ANALYSIS = "all" #Options: "thresholding", "glb_percentile", "glb_mean", "glb_both",
#  "lb_percentile", lb_mean, lb_both, "hierarchical_kmeans", adaptive_kmeans, kmeans_both, "all"
##Option for signal-to-noise and mean-to-signal ratio calculations
SNR_MSR = False # Options: True, False
PARENT_DIR = ""
# #Declare file names/patterns
fname_patterns = {'Non_corr':[r"img_ventilation\.nii\.gz$"],
    'N4_corr': [r"img_ventilation_N4\.nii\.gz$"],
    'FA_corr': [r"img_ventilation_corrected\.nii\.gz$"],
    'mask_file': [r"img_ventilation_mask\.nii\.gz$"],
    'proton_file': [r"img_proton\.nii\.gz$"],
    'proton_file_N4': [r"img_proton_N4\.nii\.gz$"]}

#%%Provide thresholds and fit parameters for glb
CORR_SEQ_ANALYSIS = [CORR_TYPE, MR_SEQUENCE, VDP_ANALYSIS, SNR_MSR]
# # Choose appropriate glb thresholds (Last Updated: 23 November 2024)
if  CORR_SEQ_ANALYSIS[1] == "Spiral":
    # #Declare the 99th-percentile normalized glb thresholds (N=25)
    percentile_glb_thresholds = {'Non_corr': [0.163995, 0.305597, 0.481527, 0.689108, 0.926391],
        'N4_corr': [0.424708, 0.5966, 0.733292, 0.850749, 0.955567],
        'FA_corr': [0.187552, 0.349312, 0.530989, 0.728438, 0.939143]}
    # #Declare the 99th-percentile glb skew fit parameters
    percentile_glb_fit = {'Non_corr': [2.30412, 0.283959, 0.285324],
        'N4_corr': [-2.206201, 0.861414, 0.193505],
        'FA_corr': [1.16688, 0.389092, 0.239586]}

    # #Declare the mean normalized glb thresholds (N=25)
    mean_glb_thresholds = {'Non_corr': [0.389662, 0.63481, 0.95643, 1.361768, 1.857616],
        'N4_corr': [0.599997, 0.82652, 1.012091, 1.174084, 1.32016],
        'FA_corr': [0.428166, 0.671352, 0.970239, 1.325215, 1.736611]}
    # #Declare the healthy cohort mean glb skew fit parameters
    mean_glb_fit = {'Non_corr': [2.85901, 0.577543, 0.562157],
        'N4_corr': [-1.904483, 1.177564, 0.253234],
        'FA_corr': [2.122713, 0.656616, 0.4758]}

elif CORR_SEQ_ANALYSIS[1] == "Cartesian":
    # #Declare the 99th-percentile normalized glb thresholds (N=11)
    percentile_glb_thresholds = {'Non_corr': [0.194855, 0.369648, 0.552419, 0.740834, 0.933658],
        'N4_corr': [0.448181, 0.621903, 0.752298, 0.860778, 0.955443]}
    #Declare the healthy cohort 99th percentile glb skew fit parameters
    percentile_glb_fit = {'Non_corr': [0.566625, 0.475872, 0.201319],
        'N4_corr': [-2.515712, 0.87849, 0.188364]}
    # #Declare the median normalized glb thresholds (N=11)
    mean_glb_thresholds = {'Non_corr': [0.364494, 0.664896, 0.989459, 1.332274, 1.689938],
        'N4_corr': [0.619569, 0.84022, 1.013677, 1.16136, 1.292169]}
    # #Declare the healthy cohort median glb skew fit parameters
    mean_glb_fit = {'Non_corr': [1.067929, 0.760544, 0.408859],
        'N4_corr': [-2.091532, 1.171888, 0.241623]}

# Combining both glb analysis methods into one dictionary each
percentile_glb_dict = {'thresholds': percentile_glb_thresholds, 'fits': percentile_glb_fit}
mean_glb_dict = {'thresholds': mean_glb_thresholds, 'fits': mean_glb_fit}
glb_dicts = [percentile_glb_dict, mean_glb_dict] # , percentile_lb_dict, mean_lb_dict

#%%Running script based on analysis mode (Single or Batch)
##Counter for Batch mode
NUM = 0
if ANALYSIS_MODE == "Single":
    # # Single subject analysis mode
    single_subj_analysis(PARENT_DIR, CORR_SEQ_ANALYSIS, fname_patterns,
                            glb_dicts)
elif ANALYSIS_MODE == "Batch":
    for dirpath, dirnames, filenames in os.walk(PARENT_DIR):
        for dirname in dirnames:
            # Check if any of the patterns match the dirname
            if any(re.match(pattern, dirname) for pattern in SUBJ_DIR_NAMES):
                print(f"Subject folder name: {dirname}")
                subject_dir=os.path.join(dirpath, dirname)
                print(f"\nSubject directory: {subject_dir}\n")
                NUM = NUM + 1
                # # #Run the function on the matching folder
                single_subj_analysis(subject_dir, CORR_SEQ_ANALYSIS, fname_patterns,
                                        glb_dicts)
    print(f"Total Subject analyzed: {NUM}")

#%% ##Define defect/hyper/snr analysis file names and headings
if  CORR_SEQ_ANALYSIS[1] == "Spiral":
    rslts_files = [('Non_corr', 'thresholding_results', 'glb-percentile_percentages',
                'glb-mean_percentages', 'hierarchical-kmeans_percentages',
                'adaptive-kmeans_percentages', 'lb-mean_percentages'),
                ('N4_corr', 'thresholding_results', 'glb-percentile_percentages',
                 'glb-mean_percentages', 'hierarchical-kmeans_percentages',
                 'adaptive-kmeans_percentages', 'lb-mean_percentages'),
                ('FA_corr', 'thresholding_results', 'glb-percentile_percentages',
                 'glb-mean_percentages', 'hierarchical-kmeans_percentages',
                 'adaptive-kmeans_percentages', 'lb-mean_percentages')] #'lb-percentile_percentages',

    vdp_headings = {'thresholding_results': ('Subject_id','VDP', 'HVP', 'NVP'),
        'glb-percentile_percentages': ('Subject_id', 'VDP', 'LVP', 'NVP1', 'NVP2','EVP', 'HVP'),
        'glb-mean_percentages': ('Subject_id', 'VDP', 'LVP', 'NVP1', 'NVP2','EVP', 'HVP'),
        'hierarchical-kmeans_percentages': ('Subject_id', 'VDP', 'LVP', 'NVP1', 'NVP2','HVP'),
        'adaptive-kmeans_percentages': ('Subject_id', 'VDR', 'LVR', 'MVR', 'HVR'),
        'lb-mean_percentages': ('Subject_id', 'VDP', 'LVP', 'NVP1', 'NVP2','EVP', 'HVP')}
    snr_data_files = [('Non_corr', 'meansig_sdbkg_overallSNR'),
              ('N4_corr', 'meansig_sdbkg_overallSNR'),
              ('FA_corr', 'meansig_sdbkg_overallSNR')]

elif CORR_SEQ_ANALYSIS[1] == "Cartesian":
    rslts_files = [('Non_corr', 'thresholding_results', 'glb-percentile_percentages',
                'glb-mean_percentages', 'hierarchical-kmeans_percentages',
                'adaptive-kmeans_percentages', 'lb-mean_percentages'),
                ('N4_corr', 'thresholding_results', 'glb-percentile_percentages',
                 'glb-mean_percentages', 'hierarchical-kmeans_percentages',
                 'adaptive-kmeans_percentages', 'lb-mean_percentages')]

    vdp_headings = {'thresholding_results': ('Subject_id','VDP', 'HVP', 'NVP'),
        'glb-percentile_percentages': ('Subject_id', 'VDP', 'LVP', 'NVP1', 'NVP2','EVP', 'HVP'),
        'glb-mean_percentages': ('Subject_id', 'VDP', 'LVP', 'NVP1', 'NVP2','EVP', 'HVP'),
        'hierarchical-kmeans_percentages': ('Subject_id', 'VDP', 'LVP', 'NVP1', 'NVP2','HVP'),
        'adaptive-kmeans_percentages': ('Subject_id', 'VDR', 'LVR', 'MVR', 'HVR'),
        'lb-mean_percentages': ('Subject_id', 'VDP', 'LVP', 'NVP1', 'NVP2','EVP', 'HVP')}
    snr_data_files = [('Non_corr', 'meansig_sdbkg_overallSNR'),
              ('N4_corr', 'meansig_sdbkg_overallSNR')]

#%% ##Gather vdp analysis results in csv
for prefix, *suffixes in rslts_files:
    for suffix in suffixes:
        text_file_name = f'{prefix}_{suffix}.txt'
        heading = vdp_headings.get(suffix, None)
        if heading:
            append_save_txt2csv(PARENT_DIR, text_file_name, heading,
                                                    correction=prefix)

#%% ##Gather SNR analysis results
snr_headings = {'meansig_sdbkg_overallSNR': ('Subject_id','mean_signal','bkgd_sd', 'snr')}
for prefix, *suffixes in snr_data_files:
    for suffix in suffixes:
        heading = snr_headings.get(suffix, None)
        if heading:
            append_save_snr_txts(PARENT_DIR, heading, correction=prefix,
                                                    analysis_type=suffix)

#%%
