# FLM 

Tensorflow implementation of "fast landmark manipulation method" (FLM) and "grouped fast landmark manipulation method" (GFLM) for generating adversarial faces, from our paper: [Fast Geometrically-Perturbed Adversarial Faces](https://arxiv.org/abs/1809.08999).

### Sample results
![](https://github.com/alldbi/FLM/blob/master/sample_results/resultss.png)

## Setup

### Prerequisites
- Tensorflow 1.4.1
- Python 2.7
- CV2
- Dlib 
- Matplotlib

### Pretrained models
- [Download](https://drive.google.com/file/d/1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz/view) pretrained "Inception ResNet v1" model, trained on the "CASIA-WebFace" dataset provided by [FaceNet](https://github.com/davidsandberg/facenet).

- [Download](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) the file and extract it to get the "shape_predictor_68_face_landmarks.dat" pretrained model for [DLib](http://dlib.net/) landmark detector.

### Getting Started

```sh
# clone this repo
git clone https://github.com/alldbi/FLM.git
cd FLM

# Generating adversarial faces by Grouped FLM:
python main.py \
  --method GFLM \
  --pretrained_model "path to the Inception ResNet v1 model trained on CASIA-WebFace" \
  --dlib_model "path to the pretrained model of the Dlib landmark detector" \
  --img "path to the input image" \
  --label "label of the input image" \
  --output_dir "path to the directory to save results"
  --epsilon "coefficient for a scaling the gradient sign for each single iteration of the attack"

# Generating adversarial faces by FLM:
python main.py \
  --method GFLM \
  --pretrained_model "path to the Inception ResNet v1 model trained on CASIA-WebFace" \
  --dlib_model "path to the pretrained model of the Dlib landmark detector" \
  --img "path to the input image" \
  --label "label of the input image" \
  --output_dir "path to the directory to save results"
  --epsilon "coefficient for a scaling the gradient sign for each single iteration of the attack"
```

## Citation
If you use this code for your research, please cite the paper: <a href="https://arxiv.org/abs/1809.08999">Fast Geometrically-Perturbed Adversarial Faces</a>:

@article{dabouei2018fast,
  title={Fast Geometrically-Perturbed Adversarial Faces},
  author={Dabouei, Ali and Soleymani, Sobhan and Dawson, Jeremy and Nasrabadi, Nasser M},
  journal={arXiv preprint arXiv:1809.08999},
  year={2018}
}

## References
- [FaceNet](https://github.com/davidsandberg/facenet)
- [TensorFlow STN implementation](https://github.com/daviddao/spatial-transformer-tensorflow/blob/master/spatial_transformer.py)
