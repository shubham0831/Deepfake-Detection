# Political Deepfakes
Generating and detecting deep-fakes of political figures

## Descriptions  
### Political Deepfakes
* [FaceSwap_GAN_v2.2_train_test.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_v2.2_train_test.ipynb)
  - Notebook for model training of faceswap-GAN model version 2.2.
  - This notebook also provides code for still image transformation at the bottom.
  - Require additional training images generated through [prep_binary_masks.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/prep_binary_masks.ipynb).
  
* [FaceSwap_GAN_v2.2_video_conversion.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_v2.2_video_conversion.ipynb)
  - Notebook for video conversion of faceswap-GAN model version 2.2.
  - Face alignment using 5-points landmarks is introduced to video conversion.
  
* [prep_binary_masks.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/prep_binary_masks.ipynb)
  - Notebook for training data preprocessing. Output binary masks are save in `./binary_masks/faceA_eyes` and `./binary_masks/faceB_eyes` folders.
  - Require [face_alignment](https://github.com/1adrianb/face-alignment) package. (An alternative method for generating binary masks (not requiring `face_alignment` and `dlib` packages) can be found in [MTCNN_video_face_detection_alignment.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/MTCNN_video_face_detection_alignment.ipynb).) 
  
* [MTCNN_video_face_detection_alignment.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/MTCNN_video_face_detection_alignment.ipynb)
  - This notebook performs face detection/alignment on the input video. 
  - Detected faces are saved in `./faces/raw_faces` and `./faces/aligned_faces` for non-aligned/aligned results respectively.
  - Crude eyes binary masks are also generated and saved in `./faces/binary_masks_eyes`. These binary masks can serve as a suboptimal alternative to masks generated through [prep_binary_masks.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/prep_binary_masks.ipynb). 
  
**Usage**
1. Run [MTCNN_video_face_detection_alignment.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/MTCNN_video_face_detection_alignment.ipynb) to extract faces from videos. The output is a series of 80x80 jpeg frames centered on the subjects' face. Manually move/rename the aligned face images into `./faceA/` or `./faceB/` folders.
2. Run [prep_binary_masks.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/prep_binary_masks.ipynb) to generate 256x256 jpeg binary masks of the eyes of training images. 
    - You can skip this pre-processing step by (1) setting `use_bm_eyes=False` in the config cell of the train_test notebook, or (2) use low-quality binary masks generated in step 1.
3. Run [FaceSwap_GAN_v2.2_train_test.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_v2.2_train_test.ipynb) to train  models. It will likely take several days to finish 40,000 iterations. If you receive an Out of Memory error, reduce the batch size (must be an even number).
4. Run  [FaceSwap_GAN_v2.2_video_conversion.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_v2.2_video_conversion.ipynb) to create videos using the trained models in step 3. 

### Training data format 
  - Face images are supposed to be in `./faceA/` or `./faceB/` folder for each target respectively. 
  - Images will be resized to 256x256 during training.

## Generative adversarial networks for face swapping
### 1. Architecture
  - GAN with a denoising Autoencoder as the generator
  - Binary classifier adversary
  - Feedforward neural network

  ![enc_arch3d](https://www.dropbox.com/s/b43x8bv5xxbo5q0/enc_arch3d_resized2.jpg?raw=1)
  
  ![dec_arch3d](https://www.dropbox.com/s/p09ioztjcxs66ey/dec_3arch3d_resized.jpg?raw=1)
  
  ![dis_arch3d](https://www.dropbox.com/s/szcq8j5axo11mu9/dis_arch3d_resized2.jpg?raw=1)

### 2. Features
- **[VGGFace](https://github.com/rcmalli/keras-vggface) perceptual loss:** Perceptual loss improves direction of eyeballs to be more realistic and consistent with input face. It also smoothes out artifacts in the segmentation mask, resulting higher output quality.

- **Attention mask:** Model predicts an attention mask that helps on handling occlusion, eliminating artifacts, and producing natrual skin tone.

- **Configurable input/output resolution (v2.2)**: The model supports 64x64, 128x128, and 256x256 outupt resolutions.

- **Face tracking/alignment using MTCNN and Kalman filter in video conversion**: 
  - MTCNN is introduced for more stable detections and reliable face alignment (FA). 
  - Kalman filter smoothen the bounding box positions over frames and eliminate jitter on the swapped face.
  
- **Eyes-aware training:** Introduce high reconstruction loss and edge loss in eyes area, which guides the model to generate realistic eyes.

## Requirements

* keras 2.1.5
* Tensorflow 1.8.0 
* Python 3.6.4
* OpenCV
* [keras-vggface](https://github.com/rcmalli/keras-vggface)
* [moviepy](http://zulko.github.io/moviepy/)
* [prefetch_generator](https://github.com/justheuristic/prefetch_generator) (required for v2.2 model)
* [face-alignment](https://github.com/1adrianb/face-alignment) (required as preprocessing for v2.2 model)

## Acknowledgments
Code borrows from [shaoanlu](https://github.com/shaoanlu/faceswap-GAN), [tjwei](https://github.com/tjwei/GANotebooks), [eriklindernoren](https://github.com/eriklindernoren/Keras-GAN/blob/master/aae/adversarial_autoencoder.py), [fchollet](https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/8.5-introduction-to-gans.ipynb), [keras-contrib](https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py) and [reddit user deepfakes' project](https://pastebin.com/hYaLNg1T). The generative network is adopted from [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). Weights and scripts of MTCNN are from [FaceNet](https://github.com/davidsandberg/facenet). Illustrations are from [irasutoya](http://www.irasutoya.com/).
Dataset: [The Presidential Deepfakes Dataset](http://ceur-ws.org/Vol-2942/paper3.pdf)
