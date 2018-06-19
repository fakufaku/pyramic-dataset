#!/bin/bash

# This will download all the segmented samples
wget -qO- https://zenodo.org/record/1209563/files/pyramic_segmented_fq_sample0.tar.gz | tar xzv
wget -qO- https://zenodo.org/record/1209563/files/pyramic_segmented_fq_sample1.tar.gz | tar xzv
wget -qO- https://zenodo.org/record/1209563/files/pyramic_segmented_fq_sample2.tar.gz | tar xzv
wget -qO- https://zenodo.org/record/1209563/files/pyramic_segmented_fq_sample3.tar.gz | tar xzv
wget -qO- https://zenodo.org/record/1209563/files/pyramic_segmented_fq_sample4.tar.gz | tar xzv
wget -qO- https://zenodo.org/record/1209563/files/pyramic_segmented_silence.tar.gz | tar xzv
wget -qO- https://zenodo.org/record/1209563/files/pyramic_segmented_sweep_exp.tar.gz | tar xzv
wget -qO- https://zenodo.org/record/1209563/files/pyramic_segmented_sweep_lin.tar.gz | tar xzv
wget -qO- https://zenodo.org/record/1209563/files/pyramic_segmented_noise.tar.gz | tar xzv
