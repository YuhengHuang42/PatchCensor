# Study of voter-like method

## File

This folder contains the experiments for voter-like method. Which includes:

1. case_study_for_CIFAR.ipynb

    - Upsampling and downsampling on CIFAR-10, includes testing their certified accuracy and clean accuracy.
    - Confusion analysis: This includes the analysis for correct but not certified instances given by the model. So we can understand what confuses the certification method.
    - Patch position invariant analysis: This includes the analysis for patch position. We first do partition and then random shuffle the patches. These new images are feed into the model and we test the performance of the model.
  
2. AOI_voter_study.ipynb

    - This file includes most of experiment results for motvation study in the paper.

3. multipleC_MNIST.ipynb

    - Enhance the difference between different labels by using RGB instead of single channel for every image. Test the certified accuracy and clean accuracy after such enhancement. 

4. dataset_preprocess.ipynb
   - Dataset preprocessing, includes: a. GTSRB. b. GTSDB. C. MNIST in three channels. 


## Other Repos

[patchGuard](https://github.com/inspire-group/PatchGuard): See the original repo for details. Here we modify the training file: *PatchGuard/misc/train_cifar.py*

[patchSmoothing](https://github.com/alevine0/patchSmoothing): See the original repo for details. Here we modify *patchSmoothing/train_cifar_band.py*



