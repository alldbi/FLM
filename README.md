# FLM 

Tensorflow implementation of "fast landmark manipulation method" (FLM) and "grouped fast landmark manipulation method" (GFLM) for generating adversarial faces, from our paper: [Fast Geometrically-Perturbed Adversarial Faces](address will be added).

## Setup

### Prerequisites
- Tensorflow 1.4.1
- CV2
- Dlib 
- Matplotlib

### Pretrained models
- [Download](https://drive.google.com/file/d/1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz/view) pretrained "Inception ResNet v1" model, trained on the "CASIA-WebFace" dataset provided by [facenet](https://github.com/davidsandberg/facenet).

- [Download](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) the file and extract it to get the "shape_predictor_68_face_landmarks.dat" pretrained model for [DLib](http://dlib.net/) landmark detector.

### Getting Started

```sh
# clone this repo
git clone https://github.com/alldbi/FLM.git
cd FLM

# Generating adversarial faces by Grouped FLM:
python main.py \
  --method GFLM \
  --output_dir facades_train \
  --max_epochs 200 \
  --input_dir facades/train \
  --which_direction BtoA
```
