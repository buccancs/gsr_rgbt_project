# GSR-RGBT Project Development Timeline

## Overview

This document provides a step-by-step timeline of the GSR-RGBT project's development, based on the repository's commit history. It shows the evolution of the project from initial scaffold to its current state, highlighting key milestones and changes.


## June 2025

### 2025-06-18 02:10:00 - Initial scaffold

**Author:** Duy An Tran  
**Commit:** f6c99ab6b6  
**Changes:** 11 files changed, 220 insertions(+), 0 deletions(-)  

**Files Changed:**

* **./**
  * .gitignore (+6, -0)
  * README.md (+3, -0)
  * build_project.py (+198, -0)
  * requirements.txt (+4, -0)
* **data/sample/**
  * sample_gsr.csv (+3, -0)
  * sample_rgb.mp4 (+1, -0)
  * sample_thermal.mp4 (+1, -0)
* **src/**
  * main.py (+1, -0)
* **src/capture/**
  * video_capture.py (+1, -0)
* **src/gui/**
  * main_window.py (+1, -0)
* **src/utils/**
  * data_logger.py (+1, -0)

---

### 2025-06-18 03:02:04 - feat: Add neurokit2, physiokit, and pyshimmer as submodules

**Author:** Duy An Tran  
**Commit:** b83c922cce  
**Changes:** 4 files changed, 12 insertions(+), 0 deletions(-)  

**Files Changed:**

* **./**
  * .gitmodules (+9, -0)
* **third_party/**
  * neurokit2 (+1, -0)
  * physiokit (+1, -0)
  * pyshimmer (+1, -0)

---

### 2025-06-18 09:22:32 - Add data collection protocol text

**Author:** Duy An Tran  
**Commit:** ce8e2ed979  
**Changes:** 4 files changed, 578 insertions(+), 0 deletions(-)  

**Files Changed:**

* **docs/**
  * appendix.tex (+295, -0)
  * data_collection_initial.tex (+84, -0)
  * data_collection_revised.tex (+65, -0)
  * proposal.tex (+134, -0)

---

### 2025-06-18 14:11:25 - Update consent form and information sheet

**Author:** Duy An Tran  
**Commit:** 234c1847a8  
**Changes:** 7 files changed, 314 insertions(+), 41 deletions(-)  

**Files Changed:**

* **docs/**
  * appendix.tex (+13, -10)
  * consent_form.tex (+129, -0)
  * data_collection_initial.tex (+14, -6)
  * data_collection_revised.tex (+25, -18)
  * information_sheet.tex (+106, -0)
  * proposal.tex (+7, -7)
  * references.bib (+20, -0)

---

### 2025-06-19 01:08:21 - added mvp

**Author:** Duy An Tran  
**Commit:** cc4173eb7e  
**Changes:** 4 files changed, 430 insertions(+), 4 deletions(-)  

**Files Changed:**

* **src/**
  * main.py (+35, -1)
* **src/capture/**
  * video_capture.py (+107, -1)
* **src/gui/**
  * main_window.py (+152, -1)
* **src/utils/**
  * data_logger.py (+136, -1)

---

### 2025-06-19 01:13:41 - added ml models

**Author:** Duy An Tran  
**Commit:** e88ad6480a  
**Changes:** 12 files changed, 43775 insertions(+), 0 deletions(-)  

**Files Changed:**

* **references/**
  * 2411.01542v1 (1).pdf (+0, -0)
  * DeepBreath_Deep_learning_of_breathing_patterns_for_automatic_stress_recognition_using_low-cost_thermal_imaging_in_unconstrained_settings.pdf (+0, -0)
  * boe-8-10-4480.pdf (+42825, -0)
  * electronics-13-01334-v2 (2).pdf (+0, -0)
  * ibvp.tex (+15, -0)
  * sensors-23-08244.pdf (+0, -0)
* **src/**
  * config.py (+50, -0)
* **src/capture/**
  * gsr_capture.py (+148, -0)
* **src/ml_models/**
  * models.py (+205, -0)
* **src/processing/**
  * data_loader.py (+154, -0)
  * feature_engineering.py (+208, -0)
  * preprocessing.py (+170, -0)

---

### 2025-06-19 01:20:23 - added updated pipeline

**Author:** Duy An Tran  
**Commit:** bc386eef8c  
**Changes:** 8 files changed, 921 insertions(+), 194 deletions(-)  

**Files Changed:**

* **./**
  * .gitignore (+40, -1)
  * Makefile (+72, -0)
  * README.md (+126, -2)
  * build_project.py (+77, -191)
* **src/evaluation/**
  * visualisation.py (+141, -0)
* **src/scripts/**
  * evaluate_model.py (+100, -0)
  * inference.py (+184, -0)
  * train_model.py (+181, -0)

---

### 2025-06-19 01:40:05 - clean up

**Author:** Duy An Tran  
**Commit:** 2d28e86954  
**Changes:** 13 files changed, 1906 insertions(+), 253 deletions(-)  

**Files Changed:**

* **./**
  * Makefile (+18, -5)
  * README.md (+142, -20)
* **docs/**
  * tasks.md (+97, -0)
* **src/ml_models/**
  * model_config.py (+275, -0)
  * models.py (+207, -49)
* **src/scripts/**
  * check_system.py (+136, -0)
  * create_mock_data.py (+185, -0)
  * inference.py (+164, -124)
  * train_model.py (+253, -55)
* **src/tests/**
  * test_data_loader.py (+106, -0)
  * test_feature_engineering.py (+125, -0)
  * test_models.py (+86, -0)
  * test_preprocessing.py (+112, -0)

---

### 2025-06-19 01:50:37 - up date mock test

**Author:** Duy An Tran  
**Commit:** 34bc608586  
**Changes:** 4 files changed, 424 insertions(+), 218 deletions(-)  

**Files Changed:**

* **src/processing/**
  * data_loader.py (+2, -0)
* **src/scripts/**
  * create_mock_data.py (+250, -110)
  * inference.py (+171, -108)
* **src/tests/**
  * test_data_loader.py (+1, -0)

---

### 2025-06-19 02:09:44 - generate testing data and chage to pytorch

**Author:** Duy An Tran  
**Commit:** ae112f9484  
**Changes:** 12 files changed, 2421 insertions(+), 189 deletions(-)  

**Files Changed:**

* **./**
  * Makefile (+15, -4)
  * README.md (+4, -2)
  * requirements.txt (+5, -0)
  * setup.py (+39, -0)
* **src/ml_models/**
  * model_config.py (+261, -35)
  * model_interface.py (+175, -0)
  * pytorch_models.py (+1342, -0)
* **src/processing/**
  * cython_optimizations.pyx (+185, -0)
  * feature_engineering.py (+87, -34)
  * preprocessing.py (+25, -5)
* **src/scripts/**
  * inference.py (+136, -53)
  * train_model.py (+147, -56)

---

### 2025-06-19 02:16:26 - update visualisation

**Author:** Duy An Tran  
**Commit:** b2de8c6f36  
**Changes:** 5 files changed, 1107 insertions(+), 4 deletions(-)  

**Files Changed:**

* **./**
  * SUMMARY.md (+114, -0)
* **src/scripts/**
  * create_mock_data.py (+9, -4)
  * generate_training_data.py (+168, -0)
  * run_ml_pipeline.py (+367, -0)
  * visualize_results.py (+449, -0)

---

### 2025-06-19 02:39:59 - added config runner and new neur networks

**Author:** Duy An Tran  
**Commit:** 514dcade87  
**Changes:** 7 files changed, 2844 insertions(+), 142 deletions(-)  

**Files Changed:**

* **./**
  * README.md (+145, -56)
* **src/ml_models/**
  * model_config.py (+73, -1)
  * pytorch_cnn_models.py (+934, -0)
  * pytorch_resnet_models.py (+550, -0)
  * pytorch_transformer_models.py (+467, -0)
* **src/scripts/**
  * run_ml_pipeline_from_config.py (+512, -0)
  * visualize_results.py (+163, -85)

---

### 2025-06-19 02:59:28 - add cnn rnn

**Author:** Duy An Tran  
**Commit:** 89153b2c4c  
**Changes:** 3 files changed, 1073 insertions(+), 168 deletions(-)  

**Files Changed:**

* **docs/**
  * assistant_logs.md (+114, -0)
* **src/ml_models/**
  * pytorch_cnn_models.py (+809, -168)
* **src/tests/**
  * test_pytorch_models.py (+150, -0)

---

### 2025-06-19 03:43:01 - update with mirror palm roi

**Author:** Duy An Tran  
**Commit:** d3bb3c4e9a  
**Changes:** 8 files changed, 904 insertions(+), 28 deletions(-)  

**Files Changed:**

* **./**
  * README.md (+25, -1)
  * requirements.txt (+2, -0)
* **docs/**
  * assistant_logs.md (+67, -0)
  * proposal.tex (+14, -14)
* **src/processing/**
  * preprocessing.py (+298, -8)
* **src/scripts/**
  * train_model.py (+205, -5)
* **src/tests/**
  * test_metadata_saving.py (+164, -0)
  * test_multi_roi.py (+129, -0)

---

### 2025-06-19 03:56:29 - add shimmer sample data and update tests

**Author:** Duy An Tran  
**Commit:** 58fc9df7f9  
**Changes:** 6 files changed, 393 insertions(+), 55 deletions(-)  

**Files Changed:**

* **docs/**
  * shimmer_integration.md (+57, -0)
* **src/**
  * config.py (+1, -1)
* **src/processing/**
  * data_loader.py (+46, -1)
  * preprocessing.py (+127, -46)
* **src/tests/**
  * test_data_loader.py (+33, -0)
  * test_preprocessing.py (+129, -7)

---

### 2025-06-19 04:53:01 - unittest and pep update

**Author:** Duy An Tran  
**Commit:** c8a6f5ea78  
**Changes:** 10 files changed, 1621 insertions(+), 47 deletions(-)  

**Files Changed:**

* **./**
  * .coverage (+0, -0)
* **src/processing/**
  * feature_engineering.py (+93, -3)
* **src/tests/**
  * test_feature_engineering.py (+190, -4)
  * test_metadata_saving.py (+19, -17)
  * test_model_config.py (+286, -0)
  * test_model_configurations.py (+319, -0)
  * test_models.py (+26, -3)
  * test_pytorch_models.py (+134, -20)
  * test_regression.py (+290, -0)
  * test_train_model.py (+264, -0)

---

### 2025-06-19 05:37:27 - unittest iteration

**Author:** Duy An Tran  
**Commit:** 8e33453901  
**Changes:** 5 files changed, 1134 insertions(+), 64 deletions(-)  

**Files Changed:**

* **src/tests/**
  * test_cython_optimizations.py (+189, -0)
  * test_model_config.py (+80, -0)
  * test_model_configurations.py (+320, -56)
  * test_pytorch_models.py (+306, -5)
  * test_regression.py (+239, -3)

---

### 2025-06-19 06:25:10 - gui preprop update and md plus tex

**Author:** Duy An Tran  
**Commit:** 4361d9e30f  
**Changes:** 8 files changed, 1117 insertions(+), 465 deletions(-)  

**Files Changed:**

* **./**
  * README.md (+79, -59)
* **docs/**
  * proposal.tex (+9, -9)
  * proposal_updated.tex (+134, -0)
* **src/capture/**
  * thermal_capture.py (+246, -0)
* **src/evaluation/**
  * real_time_visualization.py (+168, -0)
* **src/ml_models/**
  * pytorch_models.py (+392, -333)
* **src/tests/**
  * test_pytorch_models.py (+33, -6)
  * test_regression.py (+56, -58)

---

### 2025-06-19 07:32:18 - ppreprocessing and sync update

**Author:** Duy An Tran  
**Commit:** 7e43378e89  
**Changes:** 27 files changed, 3250 insertions(+), 264 deletions(-)  

**Files Changed:**

* **./**
  * .coverage (+0, -0)
  * setup.py (+7, -1)
* **docs/**
  * device_integration.md (+203, -0)
  * improvements_summary.md (+147, -0)
  * timestamp_synchronization.md (+530, -0)
* **src/**
  * config.py (+1, -0)
  * main.py (+176, -30)
* **src/capture/**
  * gsr_capture.py (+45, -43)
  * thermal_capture.py (+44, -42)
  * video_capture.py (+10, -4)
* **src/gui/**
  * main_window.py (+73, -6)
* **src/ml_models/**
  * model_config.py (+101, -18)
  * pytorch_models.py (+92, -6)
* **src/processing/**
  * cython_optimizations.pyx (+72, -32)
* **src/scripts/**
  * check_system.py (+5, -1)
  * train_model.py (+92, -29)
* **src/tests/**
  * __init__.py (+1, -0)
  * test_cython_optimizations.py (+56, -46)
* **src/tests/regression/**
  * __init__.py (+1, -0)
  * test_model_configurations.py (+563, -0)
  * test_regression.py (+269, -0)
* **src/tests/smoke/**
  * __init__.py (+1, -0)
  * test_smoke.py (+139, -0)
* **src/tests/unit/**
  * __init__.py (+1, -0)
  * test_cython_optimizations.py (+198, -0)
  * test_pytorch_models.py (+411, -0)
* **src/utils/**
  * data_logger.py (+12, -6)

---

### 2025-06-19 08:19:03 - update tex

**Author:** Duy An Tran  
**Commit:** 7d91906ccf  
**Changes:** 9 files changed, 616 insertions(+), 83 deletions(-)  

**Files Changed:**

* **docs/**
  * data_collection_initial.tex (+3, -3)
  * data_collection_revised.tex (+3, -3)
  * proposal.tex (+7, -7)
  * proposal_updated.tex (+9, -9)
  * tasks.md (+50, -50)
* **src/ml_models/**
  * pytorch_models.py (+17, -0)
* **src/scripts/**
  * compare_experiments.py (+421, -0)
  * inference.py (+24, -1)
  * train_model.py (+82, -10)

---

### 2025-06-19 08:34:43 - minor notes pieces

**Author:** Duy An Tran  
**Commit:** 7bff3f11a9  
**Changes:** 6 files changed, 492 insertions(+), 115 deletions(-)  

**Files Changed:**

* **./**
  * requirements.txt (+2, -0)
* **docs/**
  * implementation_notes.md (+169, -0)
* **references/**
  * ibvp.tex (+0, -15)
* **src/processing/**
  * feature_engineering.py (+145, -33)
* **src/scripts/**
  * compare_experiments.py (+80, -62)
* **src/utils/**
  * data_logger.py (+96, -5)

---

### 2025-06-19 09:22:45 - time sync impro and multi modal sync

**Author:** Duy An Tran  
**Commit:** ec54f41534  
**Changes:** 8 files changed, 1287 insertions(+), 23 deletions(-)  

**Files Changed:**

* **./**
  * README.md (+1, -1)
* **docs/**
  * appendix.tex (+1, -1)
  * equipment_setup.md (+594, -0)
  * implementation_improvements.md (+82, -0)
  * research_report.md (+450, -0)
* **src/**
  * main.py (+36, -3)
* **src/processing/**
  * feature_engineering.py (+63, -18)
* **src/utils/**
  * timestamp_thread.py (+60, -0)

---

