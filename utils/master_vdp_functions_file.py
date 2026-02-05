#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 11:42:06 2023

@author: Riaz Hussain, PhD
"""
import os
import sys
import warnings
import csv
import numpy as np
# import matplotlib.pyplot as plt
from skimage.morphology import cube, binary_dilation
from kmeans_analysis import hierarchical_kmeans, adaptive_kmeans
from glb_analysis import percentile_glb, mean_glb
from thresholding_analysis import mean_thresholding, save_slicewise_imgs
from mean_lb_analysis import mean_lb
import utils_funcs as ufunc
##CORR_SEQ_ANALYSIS = [CORR_TYPE, MR_SEQUENCE, VDP_ANALYSIS, SNR_MSR]
#%%Function to analyze single subject, start to finish: threshold VDP of mean & Gen lin binning
def single_subj_analysis(subj_dir, corr_seq_analisi, patterns, glb_dicts):
    """
    Performs VDP (Ventilation Defect Percentage) analysis, both threshold (e.g. 60%) of mean and
    linear binning thresholding, on a single subject's MRI data.
    Parameters
    ----------
    subj_dir (str): Path to the folder containing the MRI data of a single subject.
        The subject must have an Non_corr MRI image and a binary mask in the same folder.
    corr_seq_analisi (str): Correction, sequence type, & analysis type 
        Corrections: 'Non_corr' (Non corrected), 'N4_corr' (N4 corrected),
            'FA_corr' (FA keyhole corrected) image, or 'all' to analyze all three.
        Sequences: Cartesian or Spiral.
        Analisi: "thresholding", "glb_percentile", "glb_mean", "glb_both",
                "hierarchical_kmeans", adaptive_kmeans, kmeans_both, "lb_mean", "all"
    patterns (list, dict): File name patterns dictionary of images.
    glb_dicts (dict): generalized liear binning (glb) percentile and mean thresholds
    """
    print(subj_dir)
    # #Get correction type, subject id, correction dict, mask & proton file names and start log file
    analyses_corr_and_subj_vst, console2file, opts_dict, msk_n_prtn = init_arrangement(
                                                        subj_dir, corr_seq_analisi)
    console2file.start()
    print(f"Anaysis type/s, SNR-SRM Calc, correction/s, subject_id, visit:"
          f"{analyses_corr_and_subj_vst}")
    # Check input value and execute corresponding logic
    if analyses_corr_and_subj_vst[2] in opts_dict:
        option = opts_dict[analyses_corr_and_subj_vst[2]]
        if isinstance(option, list):
            print(f"Correction options: {option}")
            for opt in option:
                print(opts_dict[opt])
                # #Multiple correction type analysis ('all', 'corrected')
                analyses_corr_and_subj_vst[2] = opt
                print(f"\n**{opt} VDP Analysis Console output**")
                ##Dirs structure (main analysis folder + subfolders for vdp_thresh, GLB & kmeans)
                dir_lst = gen_dirs_list(subj_dir, opt,
                                        sub_dirs=analyses_corr_and_subj_vst[0]) #, new_dir=False
                img_msk_prtn = load_imgs(opt, msk_n_prtn[0], msk_n_prtn[1:], patterns, subj_dir)
                view_save_imgs(img_msk_prtn[0], img_msk_prtn[1], img_msk_prtn[2],
                                          subj_dir, opt)
                if analyses_corr_and_subj_vst[1] is True:
                    # ##Calculate and save SNR/SRM values and image maps
                    calc_save_snr_srm(analyses_corr_and_subj_vst, img_msk_prtn, dir_lst[0])
                # ##Analyze vdp/hvp as per choosen options in the main_vdp_hvp_script
                master_analysis_controller(dir_lst, img_msk_prtn, analyses_corr_and_subj_vst,
                                           glb_dicts)
            console2file.stop()
        else:
            print(option)
            # #Single correction type analysis (either of 'Non_corr', 'N4_corr', 'FA_corr')
            print(f"\n**{analyses_corr_and_subj_vst[2]} VDP Analysis Console output**")
            ##Create dir structure (main folder + subfolders for thresholding, GLB & kmeans)
            dir_lst = gen_dirs_list(subj_dir, analyses_corr_and_subj_vst[2],
                                    sub_dirs=analyses_corr_and_subj_vst[0])
            img_msk_prtn =load_imgs(analyses_corr_and_subj_vst[2], msk_n_prtn[0], msk_n_prtn[1:],
                                      patterns, subj_dir)
            view_save_imgs(img_msk_prtn[0], img_msk_prtn[1], img_msk_prtn[2],
                                      subj_dir, analyses_corr_and_subj_vst[2])
            if analyses_corr_and_subj_vst[1] is True:
                # # # Calculate and save SNR/SRM and their maps
                calc_save_snr_srm(analyses_corr_and_subj_vst, img_msk_prtn, dir_lst[0])
            # ##Analyze vdp/hvp as per choosen options in the main_vdp_hvp_script
            master_analysis_controller(dir_lst, img_msk_prtn,
                                       analyses_corr_and_subj_vst, glb_dicts)
            console2file.stop()
    else:
        # Print error message if option is not selected or is incorrect
        print("Error: Choose proper correction: ('Non_corr', 'N4_corr', 'FA_corr', or 'all')")
        console2file.stop()

def master_analysis_controller(dirs, img_msk_prtn, analisi_corr_subjid, glb_dicts):
    """Master function to choose what type of analysis to perform
    Options: "thresholding", "glb_percentile", "glb_mean", "glb_both",
            "hierarchical_kmeans", "adaptive_kmeans", "kmeans_both"
            "lb_mean", "all" (to run all 6 of them)
    """
    defect_analysis,_ , correc , _, _ = analisi_corr_subjid

    if defect_analysis == "thresholding":
        thresholding_analysis(correc, dirs, img_msk_prtn, analisi_corr_subjid)
    elif defect_analysis in ["glb_percentile", "glb_mean", "glb_both"]:
        glb_analysis(defect_analysis, dirs, img_msk_prtn, glb_dicts, analisi_corr_subjid)
    elif defect_analysis in ["hierarchical_kmeans", "adaptive_kmeans", "kmeans_both"]:
        kmeans_analysis(defect_analysis, dirs, img_msk_prtn, analisi_corr_subjid)
    elif defect_analysis == "lb_mean":
        mean_lb_analysis(defect_analysis, dirs, img_msk_prtn, analisi_corr_subjid)
    elif defect_analysis == "all":
        analysis_all(defect_analysis, dirs, img_msk_prtn, glb_dicts, analisi_corr_subjid)
    else:
        raise ValueError("Only following options are accepted:\n"
                         "thresholding, glb_percentile, glb_mean, glb_both\n"
                         "hierarchical_kmeans, adaptive_kmeans, kmeans_both\n"
                         "lb_mean, all")

def thresholding_analysis(correction, directories, im_msk_prtn, analisi_corr_subj):
    """To run thresjolding analysis"""
    mskd_img = im_msk_prtn[0] * im_msk_prtn[1]
    mskd_img /= np.max(mskd_img)
    # # ##Calculate mean--thresholded VDP/HVP
    if correction == "FA_corr":
        vdp_hvp_thresholds = [40, 200]
        print("FA correction: Using 40 percent of mean-thresholding")
    else:
        vdp_hvp_thresholds = [60, 200]
        print("Using 60 percent of mean-thresholding")
    print("Only running Thresholding analysis")
    mean_thresholding(directories[1], mskd_img, im_msk_prtn[1],
                        vdp_hvp_thresholds, analisi_corr_subj)

def glb_analysis(defect_analysis, directories, im_msk_prtn, glb_dicts, analisi_corr_subj):
    """To run generalized linear binning analysis options"""
        # # ##Calculate generalized linear binning VDP/HVP - percentile
    if defect_analysis == "glb_percentile":
        print(f"Only running {defect_analysis} analysis")
        percentile_glb(im_msk_prtn, directories[2], glb_dicts[0],
                                analisi_corr_subj)
    # # ##Calculate generalized linear binning VDP/HVP - mean
    elif defect_analysis == "glb_mean":
        print(f"Only running {defect_analysis} analysis")
        mean_glb(im_msk_prtn, directories[2], glb_dicts[1],
                            analisi_corr_subj)
    # # ##Calculate generalized linear binning VDP/HVP - both
    elif defect_analysis == "glb_both":
        print(f"Running {defect_analysis} analyses")
        percentile_glb(im_msk_prtn, directories[2], glb_dicts[0],
                                analisi_corr_subj)
        mean_glb(im_msk_prtn, directories[2], glb_dicts[1],
                            analisi_corr_subj)

def kmeans_analysis(defect_analysis, directories, im_msk_prtn, analisi_corr_subj):
    """To run k-means analysis options"""
        # # ##Calculate K-means clustering VDP/HVP - hierarchical
    if defect_analysis == "hierarchical_kmeans":
        print(f"Only running {defect_analysis} analysis")
        hierarchical_kmeans(im_msk_prtn, directories[3], analisi_corr_subj)
    # # ##Calculate K-means clustering VDP/HVP - adaptive
    elif defect_analysis == "adaptive_kmeans":
        print(f"Only running {defect_analysis} analysis")
        adaptive_kmeans(im_msk_prtn, directories[3], analisi_corr_subj)
    # # ##Calculate K-means clustering VDP/HVP - both
    elif defect_analysis == "kmeans_both":
        print(f"Running {defect_analysis} analyses")
        hierarchical_kmeans(im_msk_prtn, directories[3], analisi_corr_subj)
        adaptive_kmeans(im_msk_prtn, directories[3], analisi_corr_subj)

def mean_lb_analysis(defect_analysis, directories, im_msk_prtn, analisi_corr_subj):
    """To run mean normalized linear binning analysis options"""
    # # ##Calculate generalized linear binning VDP/HVP - mean
    print(f"Only running {defect_analysis} analysis")
    mean_lb(im_msk_prtn, directories[4], analisi_corr_subj)

def analysis_all(defect_opt, directories, im_msk_prtn, glb_dicts, analisi_corr_subj):
    """To run all analysis option"""
    print(f"Running {defect_opt} the analyses")
    mskd_norm_img = im_msk_prtn[0] * im_msk_prtn[1]
    mskd_norm_img /= np.max(mskd_norm_img)
    if analisi_corr_subj[2] == "FA_corr":
        vdp_hvp_thresholds = [40, 200]
        print("FA correction: Using 40 percent of mean-thresholding")
    else:
        vdp_hvp_thresholds = [60, 200]
        print("Using 60 percent of mean-thresholding")
    mean_thresholding(directories[1], mskd_norm_img, im_msk_prtn[1], vdp_hvp_thresholds,
                            analisi_corr_subj)
    percentile_glb(im_msk_prtn, directories[2], glb_dicts[0],
                            analisi_corr_subj)
    mean_glb(im_msk_prtn, directories[2], glb_dicts[1],
                        analisi_corr_subj)
    hierarchical_kmeans(im_msk_prtn, directories[3], analisi_corr_subj)
    adaptive_kmeans(im_msk_prtn, directories[3], analisi_corr_subj)
    mean_lb(im_msk_prtn, directories[4], analisi_corr_subj)

def gen_dirs_list(subj_dir, dtype, new_dir=False, sub_dirs='all'):
    """
    Create and return a list of appropriate directories: main, threshold, glb, kmeans.

    Returns:
        list: A list of directories in the order: [main_dir, thresh_dir, glb_dirs, kmeans_dirs].
              Entries for directories not requested by `sub_dirs` will be `None`.
    """
    # Call create_subdirs and unpack the return value
    main_dir, thresh_dir, glb_dirs, kmeans_dirs, lb_dirs = create_subdirs(
                        subj_dir, dtype, create_new=new_dir, sub_dirs=sub_dirs)
    # Create the list with the directories
    dirs_list = [main_dir, thresh_dir, glb_dirs, kmeans_dirs, lb_dirs]
    return dirs_list

def init_arrangement(subj_dir, corr_seq_analysis):
    """Initial arrangement of analysis options"""
    correc_name, seq_name, vdp_analisi, snr_srm = corr_seq_analysis
    vdp_log_file_path = os.path.join(subj_dir, f"{correc_name}_vdp_analysis_log.txt")
    console2f = ufunc.ConsoleToFile(vdp_log_file_path)
    if 'visit' in os.path.basename(subj_dir):
        subj_id = os.path.basename(os.path.dirname(subj_dir))
        visit_id = os.path.basename(subj_dir)
        analyses_correc_subid_visid = [vdp_analisi, snr_srm, correc_name, subj_id, visit_id]
    else:
        subj_id = os.path.basename(subj_dir)
        analyses_correc_subid_visid = [vdp_analisi, snr_srm, correc_name, subj_id, None]
    # subj_id = os.path.basename(subj_dir)
    print(f"\nThe subject id is: {subj_id}")
    msk_prtn_files = ['mask_file', 'proton_file_N4', 'proton_file']
    # #Choose appropriate names for sequence
    if seq_name == 'Spiral':
        opts = ['Non_corr', 'N4_corr', 'FA_corr']
        # Define a dictionary to map input values to appropriate action
        opts_dict = {'all': [opts[0], opts[1], opts[2]],
                     'corrected' : [opts[1], opts[2]],
            opts[0]: '\nAnalyzing uncorrected (Non_corr) image\n',
            opts[1]: '\nAnalyzing N4 corrected image\n',
            opts[2]: '\nAnalyzing FA corrected image\n'}
    elif seq_name == 'Cartesian':
        opts = ['Non_corr', 'N4_corr']
        # Define a dictionary to map input values to appropriate action
        opts_dict = {'all': [opts[0], opts[1]],
                     'corrected' : [opts[1]],
            opts[0]: '\nAnalyzing uncorrected (Non_corr) image\n',
            opts[1]: '\nAnalyzing N4 corrected image\n'}
    else:
        sys.exit("Only 'Spiral' or 'Cartesian' options accepted.")
    return analyses_correc_subid_visid, console2f, opts_dict, msk_prtn_files

def load_imgs(img_name, msk_name, prtn_name, patterns, sub_dir):
    """To load MR image, binary mask and proton nifty files"""
    #Loading MRI data
    img_data = ufunc.process_nifti_file(img_name, patterns, sub_dir)
    print("Image data dims: ", np.shape(img_data))
    #Loading binary MR mask
    msk_data = ufunc.process_nifti_file(msk_name, patterns, sub_dir)
    if img_data.shape == msk_data.shape:
        print("Image and mask arrays are equal.")
    else:
        raise ValueError(f"Image and mask arrays are not equal.\nPlease check {sub_dir}")
    prtn_n4, prtn = prtn_name
    #Loading proton if provided
    try:
        # Try to load the N4-corrected proton file
        prtn_data = ufunc.process_nifti_file(prtn_n4, patterns, sub_dir)
        if prtn_data.any():
            print("\nN4-corrected proton found.")
    except FileNotFoundError:
        # If N4-corrected proton not found, try without correction
        try:
            prtn_data = ufunc.process_nifti_file(prtn, patterns, sub_dir)
            print("\nN4-corrected proton not found.\nLoading uncorrected proton file")
        except FileNotFoundError:
            # If neither N4-corrected nor uncorrected proton file is found
            print("\nProton file not found.")
            prtn_data = None
    if prtn_data is not None and np.any(prtn_data):
        print("\nProton data dims (pre-matching): ", np.shape(prtn_data))
        prtn_data = ufunc.match_img_dimensions(img_data, prtn_data)
        mskd_prtn = ufunc.create_montage_array(prtn_data, 'all', msk_data)
    else:
        mskd_prtn = None
    return img_data, msk_data, mskd_prtn

def view_save_imgs(img_data, msk_data, prtn_data, sv_dir, data_type):
    """Function to plot MR image, binary mask and proton nifty files"""
    mskd_img = img_data * msk_data
    mskd_norm_img = mskd_img/np.max(mskd_img)
    ##Create directory for saving initial images
    save_dir = os.path.join(sv_dir, 'z_mr_images', f"{data_type}_imgs")
    os.makedirs(save_dir, exist_ok=True)
    save_slicewise_imgs(img_data, save_dir, data_type, im_type='raw')
    ##Plot and save MR image (without mask) montage
    raw_img_montage_array=ufunc.create_montage_array(img_data, slices='all')
    raw_img_montage=ufunc.montage_plot_3d(raw_img_montage_array)
    ufunc.save_image(raw_img_montage, save_dir, data_type, 'raw_montage')
    ##Plot and save binary mask
    msk_montage_array=ufunc.create_montage_array(msk_data, 'all', msk_data)
    msk_montage=ufunc.montage_plot_3d(msk_montage_array)
    ufunc.save_image(msk_montage, save_dir, data_type, 'mask_montage')
    ##Plot and save masked MR image
    mskd_img_montage_array=ufunc.create_montage_array(mskd_norm_img, 'all', msk_data)
    mskd_img_montage=ufunc.montage_plot_3d(mskd_img_montage_array)
    ufunc.save_image(mskd_img_montage, save_dir, data_type, 'masked_montage')
    ##Plot proton image (without mask) montage if present
    if prtn_data is not None:
        raw_prtn_montage=ufunc.montage_plot_3d(prtn_data)
        ufunc.save_image(raw_prtn_montage, save_dir, data_type, 'proton_montage')

def create_subdirs(subj_dir, data_type=None, create_new=False, sub_dirs='all'):
    """
    Creates directories inside 'subj_dir' and returns their paths.

    Returns:
        tuple: (main_dir, thresh_dir, glb_dirs, kmeans_dirs)
               Any non-applicable directory path will be `None`.
    """
    dir_name = "vdp_analysis_misc" if data_type is None else f"{data_type}_vdp_analysis"
    dir_vdp_analysis = os.path.join(subj_dir, dir_name)

    dir_vdp_analysis = main_direc(dir_vdp_analysis, dir_name, subj_dir, create_new)
    thresholding_dir = None
    dirs_glb = None
    dirs_kmeans = None
    dirs_lb = None
    if sub_dirs == 'thresholding':
        thresholding_dir = os.path.join(dir_vdp_analysis, "thresholding_analysis")
        os.makedirs(thresholding_dir, exist_ok=True)
    elif sub_dirs in ['glb_percentile', 'glb_mean', 'glb_both']:
        dirs_glb = create_glb_dirs(dir_vdp_analysis, glb=sub_dirs.split('_')[1])
    elif sub_dirs in ['hierarchical_kmeans', 'adaptive_kmeans', 'kmeans_both']:
        dirs_kmeans = create_kmeans_dirs(dir_vdp_analysis, kmeans=sub_dirs.split('_')[0])
    elif sub_dirs == 'lb_mean':
        dirs_lb = create_lb_dirs(dir_vdp_analysis)
    elif sub_dirs == 'all':
        thresholding_dir = os.path.join(dir_vdp_analysis, "thresholding_analysis")
        os.makedirs(thresholding_dir, exist_ok=True)
        dirs_glb = create_glb_dirs(dir_vdp_analysis)
        dirs_kmeans = create_kmeans_dirs(dir_vdp_analysis)
        dirs_lb = create_lb_dirs(dir_vdp_analysis)
    return dir_vdp_analysis, thresholding_dir, dirs_glb, dirs_kmeans, dirs_lb

def main_direc(vdp_dir, dir_name, subj_dir, create_new=False):
    """To create main directory"""
    if os.path.isdir(vdp_dir):
        if create_new:
            print(f"\n{vdp_dir} already exists. Creating new with subdirs")
            i = 1
            while True:
                new_dir_name = f"{dir_name}_{i}"
                new_dir_vdp_analysis = os.path.join(subj_dir, new_dir_name)
                if not os.path.isdir(new_dir_vdp_analysis):
                    os.makedirs(new_dir_vdp_analysis, exist_ok=True)
                    print(f"\n{new_dir_vdp_analysis} created with subdirs")
                    vdp_dir = new_dir_vdp_analysis
                    break
                i += 1
        else:
            print(f"\n{vdp_dir} already exists.")
    else:
        os.makedirs(vdp_dir, exist_ok=True)
        print(f"\nCreating dir: {vdp_dir} with subdir/s")
    return vdp_dir

def create_glb_dirs(analysis_dir, glb='both'):
    """To create generalized linear binning sub-directories"""
    # Create the "glb_analysis_percentile" directory inside {data_type}_vdp_hvp_analysis
    if glb == 'percentile':
        dir_glb_percentile = os.path.join(analysis_dir, "glb_percentile_analysis")
        os.makedirs(dir_glb_percentile, exist_ok=True)
        dirs_glb = dir_glb_percentile
    elif glb=='mean':
        dir_glb_mean = os.path.join(analysis_dir, "glb_mean_analysis")
        os.makedirs(dir_glb_mean, exist_ok=True)
        dirs_glb = dir_glb_mean
    else:
        dir_glb_percentile = os.path.join(analysis_dir, "glb_percentile_analysis")
        # Create the "GLB_analysis_mean" directory inside {data_type}_vdp_hvp_analysis
        dir_glb_mean = os.path.join(analysis_dir, "glb_mean_analysis")
        os.makedirs(dir_glb_percentile, exist_ok=True)
        os.makedirs(dir_glb_mean, exist_ok=True)
        dirs_glb = [dir_glb_percentile, dir_glb_mean]
    return dirs_glb

def create_kmeans_dirs(analysis_dir, kmeans='both'):
    """To create kmeans sub-directories"""
    if kmeans == 'hierarchical_kmeans':
        dir_kmeans_hrchy = os.path.join(analysis_dir, "kmeans_hierarchical_analysis")
        os.makedirs(dir_kmeans_hrchy, exist_ok=True)
        dirs_kmeans = dir_kmeans_hrchy
    elif kmeans == 'adaptive_kmeans':
        dir_kmeans_adpt = os.path.join(analysis_dir, "kmeans_adaptive_analysis")
        os.makedirs(dir_kmeans_adpt, exist_ok=True)
        dirs_kmeans = dir_kmeans_adpt
    else:
        # Create the "hierarchy_kmeans" directory inside {data_type}_vdp_hvp_analysis
        dir_kmeans_hrchy = os.path.join(analysis_dir, "kmeans_hierarchical_analysis")
        # Create the "adaptive_kmeans" directory inside {data_type}_vdp_hvp_analysis
        dir_kmeans_adpt = os.path.join(analysis_dir, "kmeans_adaptive_analysis")
        os.makedirs(dir_kmeans_hrchy, exist_ok=True)
        os.makedirs(dir_kmeans_adpt, exist_ok=True)
        dirs_kmeans = [dir_kmeans_hrchy, dir_kmeans_adpt]
    return dirs_kmeans

def create_lb_dirs(analysis_dir):
    """To create generalized linear binning sub-directories"""
    # Create the "glb_analysis_percentile" directory inside {data_type}_vdp_hvp_analysis
    dir_lb_mean = os.path.join(analysis_dir, "lb_mean_analysis")
    os.makedirs(dir_lb_mean, exist_ok=True)
    dirs_lb = dir_lb_mean
    return dirs_lb

def calc_save_snr_srm(analisi_corr_e_subjid, img_msk_protn, directry):
    """Master SNR/SRM calculation function"""
    _, _, corr_type, subject_id, _ = analisi_corr_e_subjid
    img, msk, prtn = img_msk_protn
    mskd_norm_img = (img * msk) / np.max(img * msk)
    mskd_norm_img[msk == 0] = np.nan
    # # ##Calculate signal-to-noise ratio (SNR) of the image
    voxelwise_snr_image=calculate_snr(img, msk, directry, corr_type, subject_id)
    save_snr_map(voxelwise_snr_image, msk, prtn, directry, corr_type)
    # # ##Calculate mean-to-signal ratio (SRM) of MR image
    ufunc.calc_slicewise_srm(mskd_norm_img, directry, subject_id, corr_type, param='mean')
    srm_montage_arr = ufunc.create_srm_array(img, msk, param='mean')
    srm_fig = ufunc.montage_plot_3d(srm_montage_arr, prtn, clr=True)
    ufunc.save_image(srm_fig, directry, corr_type, 'srm_map')

def calculate_snr(mr_image, mask_data, save_path,
                    f_name=None, subject_id=None):
    """
    Calculate signal-to-noise ratio (SNR) for normalized masked MR data.

    Parameters
    ----------
    mr_image (ndarray): Numpy array of the MR data.
    mask_data (ndarray): Numpy array of the mask data.
    save_path (str): Path to the directory where snr files should be saved.
    f_name (str, optional): Output file name of the MR data.
        If not provided, uses 'mr_image' followed by _ and slicewiseSNR.txt or OverallSNR.txt
        If file already exists, adds _2 ... to the file name.
    subject_id (str, optional): if provided appends the subject name with SNR files
    """
    print("\nCalculating normalized SNR\n")
    snr_results = os.path.abspath(save_path)
    print(f"\nSaving SNR files in: {snr_results}")
    mr_image /= np.max(mr_image)
    masked_mr_image, norm_noise = prepare_snr(mask_data, mr_image)
    slicewise_file, overall_f_name = snr_fnames(snr_results, subject_id, f_name)
    slicew_snr = calc_slicewise_snr(mr_image, masked_mr_image, norm_noise)
    #arrange and save overall snr results
    overall_sd, overall_mean_sd_snr, norm_hdng = calc_overall_snr(masked_mr_image,
                                                                  norm_noise)
    ufunc.save_txt_file(overall_mean_sd_snr, save_path, overall_f_name,
                        norm_hdng, subject_id)
    #arrange and save slice-wise snr results
    save_slicewise_txt(slicewise_file, slicew_snr)
    voxelwise_snr_image = (masked_mr_image/overall_sd) * np.sqrt(2-(np.pi/2))
    return voxelwise_snr_image

def prepare_snr(msk, mr_img):
    """Get dilated mask and background array"""
    mask_dilated = binary_dilation(msk, footprint=cube(28))
    # Take the complement of the dilated mask array and create a background noise array
    background_norm = mr_img * (np.array(mask_dilated == 0, dtype=np.uint8))
    # Label 0s in the normalized background noise array as NaNs
    background_norm[background_norm == 0] = np.nan
    # Label 0s in the normalized MR array as NaNs
    masked_mr_image = mr_img * (msk > 0)
    masked_mr_image[masked_mr_image == 0] = np.nan
    return masked_mr_image, background_norm

def snr_fnames(snr_dir, subj_id, f_name=None):
    """Generate file names for snr files"""
    ## Save the overall average SNR and the slice-wise SNR values in text files
    if f_name is None:
        f_name = f"{subj_id}"
        norm_slicewise_f_name = f"{f_name}_slicewiseSNR.csv"
        norm_overall_f_name = f"{f_name}_meansig_sdbkg_overallSNR.txt"
    else:
        norm_slicewise_f_name = f"{subj_id}_{os.path.splitext(f_name)[0]}_slicewiseSNR.csv"
        norm_overall_f_name = f"{f_name}_meansig_sdbkg_overallSNR.txt"
    i = 1
    while os.path.exists(os.path.join(snr_dir, norm_slicewise_f_name)) or \
            os.path.exists(os.path.join(snr_dir, norm_overall_f_name)):
        i += 1
        norm_slicewise_f_name = f"{subj_id}_{f_name}_slicewiseSNR_{i}.csv"
        norm_overall_f_name = f"{f_name}_overallSNR_meansig_sdbkg_{i}.txt"
    # Define the output file path
    slicewise_fname = os.path.join(snr_dir, norm_slicewise_f_name)
    return slicewise_fname, norm_overall_f_name

def calc_slicewise_snr(mr_image, msked_norm_image, bkgd_norm):
    """calculate and save slice-wise SNR"""
    slicewise_norm_snr = []
    for a_slice in range(mr_image.shape[2]):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                norm_snr_numerator = np.nanmean(msked_norm_image[:, :, a_slice])
                norm_snr_denominator = np.nanstd(bkgd_norm[:, :, a_slice])
                norm_snr_r = (norm_snr_numerator / norm_snr_denominator) * np.sqrt(2-(np.pi/2))
                mask = ~np.isnan(norm_snr_r)
                slicewise_norm_mean_sd_snr = [norm_snr_numerator[mask], norm_snr_denominator[mask],
                                               norm_snr_r[mask]]
            slicewise_norm_snr.append(slicewise_norm_mean_sd_snr)
        except ValueError as err:
            print(f"Error calculating normalized SNR for slice {a_slice}: {err}")
            continue
    return  slicewise_norm_snr

def calc_overall_snr(msked_img, norm_bkgrnd):
    """Calculate overall snr"""
    # #Calculate the overall average SNR
    norm_mean_signal = np.nanmean(msked_img)
    overall_norm_sd = np.nanstd(norm_bkgrnd)
    norm_overall_snr = (norm_mean_signal / overall_norm_sd) * np.sqrt(2-(np.pi/2))
    overall_norm_mean_sd_snr = [norm_mean_signal, overall_norm_sd, norm_overall_snr]
    norm_names = ["mean_signal", "bkgd_sd", "snr"]
    return overall_norm_sd, overall_norm_mean_sd_snr, norm_names

def save_snr_map(voxelwise_snr_image, msk_data, prtn, snr_results, f_name):
    """Plot and save snr montage"""
    snr_montage_array=ufunc.create_montage_array(voxelwise_snr_image, 'all', msk_data)
    snr_montage_array = (snr_montage_array - np.nanmin(snr_montage_array)
                     ) / (np.nanmax(snr_montage_array) - np.nanmin(snr_montage_array))
    snr_montage=ufunc.montage_plot_3d(snr_montage_array, prtn, clr=True)
    ufunc.save_image(snr_montage, snr_results, data_type=f_name, im_type='snr_map')

def save_slicewise_txt(slwise_fname, slice_snr_vals):
    """Save slice-wise snr results as text file"""
    with open(slwise_fname, mode='w', newline='', encoding="utf-8") as output_file:
        # Create a CSV writer object
        writer = csv.writer(output_file)
        # Write the header row
        writer.writerow(["mean_signal", "bkgd_sd", "snr"])
        # Write the data rows
        for row in slice_snr_vals:
            # Skip rows with empty lists
            if not row:
                continue
            # Convert values to strings and remove braces
            row_str = [str(val).strip('[]') for val in row]
            if not all(x == '' for x in row_str):
                # Write the row to the CSV file
                writer.writerow(row_str)

#%%
