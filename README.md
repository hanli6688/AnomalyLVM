# AnomalyLVM
AnomalyLVM:Vision-Language Models for Zero-Shot Anomaly Detection
# Introduction
This is the official data link of AnomalyLVM
# datasets

Industrial Visual Anomaly Detection Datasets:

MVTec AD https://drive.google.com/file/d/12IukAqxOj497J4F0Mel-FvaONM030qwP/view?usp=drive_link

VisA https://drive.google.com/file/d/1U0MZVro5yGgaHNQ8kWb3U1a0Qlz4HiHI/view?usp=drive_link

MPDD https://drive.google.com/file/d/1cLkZs8pN8onQzfyNskeU_836JLjrtJz1/view?usp=drive_link

BTAD https://drive.google.com/file/d/19Kd8jJLxZExwiTc9__6_r_jPqkmTXt4h/view?usp=drive_link

KSDD https://drive.google.com/file/d/13UidsM1taqEAVV_JJTBiCV1D3KUBpmpj/view?usp=drive_link

DTD-Synthetic https://drive.google.com/file/d/1em51XXz5_aBNRJlJxxv3-Ed1dO9H3QgS/view?usp=drive_link

Medical Visual Anomaly Detection Datasets:

HeadCT https://drive.google.com/file/d/1ore0yCV31oLwwC--YUuTQfij-f2V32O2/view?usp=drive_link

BrainMRI https://drive.google.com/file/d/1JLYyzcPG3ULY2J_aw1SY9esNujYm9GKd/view?usp=drive_link

Br35H https://drive.google.com/file/d/1qaZ6VJDRk3Ix3oVp3NpFyTsqXLJ_JjQy/view?usp=drive_link

ISIC https://drive.google.com/file/d/1atZwmnFsz7mCsHWBZ8pkL_-Eul9bKFEx/view?usp=drive_link

ColonDB https://drive.google.com/file/d/1tjZ0o5dgzka3wf_p4ErSRJ9fcC-RJK8R/view?usp=drive_link

ClinicDB https://drive.google.com/file/d/1ciqZwMs1smSGDlwQ6tsr6YzylrqQBn9n/view?usp=drive_link

TN3K https://drive.google.com/file/d/1LuKEMhrUGwFBlGCaej46WoooH89V3O8_/view?usp=drive_link
# Content

We provide the full implementation and benchmark results for AnomalyLVM, a unified zero-shot anomaly detection framework capable of performing both image-level classification and pixel-level localization.

The model utilizes frozen SAM2 and DINO-X encoders for spatial and semantic feature extraction, which are fused and decoded to generate pixel-level anomaly maps. A frozen CLIP text encoder is used for semantic similarity–based image-level detection.

The Feature Enhancement Module (FEM) refines static LayerCAM maps by combining dual-encoder attention cues and decoder affinity relations, producing more reliable and context-aware pseudo labels. An object-agnostic prompt design is adopted to eliminate handcrafted templates and ensure cross-category and cross-domain generalization.
