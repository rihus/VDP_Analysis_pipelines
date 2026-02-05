#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 7:45:48 2023

@author: Riaz Hussain, PhD

Please see/reference this paper:
Riaz Hussain, Abdullah S. Bdaiwi, Joseph W. Plummer, Bilal I. Masokano, Matthew M. Willmering, Laura L. Walkup, Zackary I. Cleveland,
Comparing Mean-anchored Generalized Linear Binning with Established Methods to Quantify Xenon-129 Ventilation Defect Percentage,
Academic Radiology, Volume 33, Issue 2, 2026, Pages 569-585, ISSN 1076-6332,
https://doi.org/10.1016/j.acra.2025.10.044.
(https://www.sciencedirect.com/science/article/pii/S107663322501030X)

This script allows six types of analysis for hyperpolarized 129Xe ventilation MR images:
    1. Hierarchical K-means clustering (Kirby et al., Acad Radiol 2012)
    2. Adaptive K-means clustering (Zha et al., Acad Radiol 2016 and 2018) 
    3. The 60% of the total image intesity mean thresholding (Thomen et al., J. of CF 2017)
    4. Generalized linear binning - 99th Percentile normalized (He et al., Acad Radiol 2020)
    5. Linear binning - mean normalized [fixed thresholds] (Collier et al, ISMRM 2020) 
    6. Generalized linear binning - Mean normalized (Hussain et al., Acad Radiol 2026)

    Last updated: Feb 5 2026

"""
#%%Import Libraries and import master vdp_hvp functions file
import os
import re
#Import file with all the vdp functions
from utils.master_vdp_functions_file import single_subj_analysis

#%%Decalre analysis mode, subject directory names, MR Sequence type,
## Bias correction type, types of VDP/HVP analysis, SNR/MSR switch,
## Main directory, File name patterns, and thresholds
ANALYSIS_MODE = "Single" # Options: "Single", "Batch"
MR_SEQUENCE = "Spiral" # Options: "Cartesian", "Spiral"
##Provide directory name/s containing subject data (pattern or full names list)
##Batch processing only
SUBJ_DIR_NAMES = ["sub_001", "sub_002", "sub_003"]
##Select bias field correction
CORR_TYPE = "corrected" # Options: "Non_corr", "N4_corr", "FA_corr" (only for Spiral), "corrected", "all"
##VDP analysis correction
VDP_ANALYSIS = "all" #Options: "thresholding", "glb_percentile", "glb_mean", "glb_both",
#  "lb_percentile", lb_mean, lb_both, "hierarchical_kmeans", adaptive_kmeans, kmeans_both, "all"
##Option for signal-to-noise and mean-to-signal ratio calculations
SNR_MSR = False # Options: True, False
##Main directory containing all subjects data - default: cwd/example_data
PARENT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "example_data")
# #Declare file names/patterns [here I'm using phantom images]
fname_patterns = {'Non_corr':[r"phantom_129Xe\.nii\.gz$"],
    'N4_corr': [r"phantom_129Xe_N4\.nii\.gz$"],
    'FA_corr': [r"phantom_129Xe_FA\.nii\.gz$"],
    'mask_file': [r"phantom_129Xe_mask\.nii\.gz$"],
    'proton_file': [r"phantom_1H\.nii\.gz$"],
    'proton_file_N4': [r"phantom_1H_N4\.nii\.gz$"]}

#%%Provide thresholds and fit parameters for glb analysis
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

#%%
