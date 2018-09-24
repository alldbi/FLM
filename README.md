# FLM 

Tensorflow implementation of "fast landmark manipulation method" (FLM) and "grouped fast landmark manipulation method" (GFLM) for generating adversarial faces, from our paper: [Fast Geometrically-Perturbed Adversarial Faces](address to the file).

## Setup

### Prerequisites
- Tensorflow 1.4.1
- Dlib 
- CV2

### Pretrained models
- [Download](https://drive.google.com/file/d/1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz/view) pretrained "Inception ResNet v1" model, trained on the "CASIA-WebFace" dataset provided by [facenet](https://github.com/davidsandberg/facenet).

- [Download](https://drive.google.com/file/d/1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz/view) and extract to get the "shape_predictor_68_face_landmarks.dat" pretrained model for [DLib](http://dlib.net/) landmark detector.

