# FLM 

Tensorflow implementation of "fast landmark manipulation method" (FLM) and "grouped fast landmark manipulation method" (GFLM) for generating adversarial faces, from our paper: [Fast Geometrically-Perturbed Adversarial Faces](address will be added).

## Setup

### Prerequisites
- Tensorflow 1.4.1
- Dlib 
- CV2

### Pretrained models
- [Download](https://drive.google.com/file/d/1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz/view) pretrained "Inception ResNet v1" model, trained on the "CASIA-WebFace" dataset provided by [facenet](https://github.com/davidsandberg/facenet).

- [Download](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) the file and extract it to get the "shape_predictor_68_face_landmarks.dat" pretrained model for [DLib](http://dlib.net/) landmark detector.

### Generating adversarial faces

```sh
# clone this repo
git clone https://github.com/affinelayer/pix2pix-tensorflow.git
cd pix2pix-tensorflow
# download the CMP Facades dataset (generated from http://cmp.felk.cvut.cz/~tylecr1/facade/)
python tools/download-dataset.py facades
# train the model (this may take 1-8 hours depending on GPU, on CPU you will be waiting for a bit)
python pix2pix.py \
  --mode train \
  --output_dir facades_train \
  --max_epochs 200 \
  --input_dir facades/train \
  --which_direction BtoA
# test the model
python pix2pix.py \
  --mode test \
  --output_dir facades_test \
  --input_dir facades/val \
  --checkpoint facades_train
