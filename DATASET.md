# Pose with Style: human reposing with pose-guided StyleGAN2


## Dataset and Downloads
1. Download images:
   1. Download `img_highres.zip` from [In-shop Clothes Retrieval Benchmark](https://drive.google.com/drive/folders/0B7EVK8r0v71pYkd5TzBiclMzR00?resourcekey=0-fsjVShvqXP2517KnwaZ0zw). You will need to follow the [download instructions](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) to unzip the file. Unzip file in `DATASET/DeepFashion_highres/img_highres`

   2. Download the [train/test data](https://drive.google.com/drive/folders/1BX3Bxh8KG01yKWViRY0WTyDWbJHju-SL): **train.lst**, **test.lst**, and **fashion-pairs-test.csv**. Put in `DATASET/DeepFashion_highres/tools`. Note: because not all training images had their densepose detected we used a slightly modified training pairs file [**fashion-pairs-test.csv**](https://filebox.ece.vt.edu/~Badour/pose-with-style/downloads/fashion-pairs-train.csv).

   3. Split the train/test dataset using:
      ```
      python util/generate_fashion_datasets.py --dataroot DATASET/DeepFashion_highres
      ```
      This will save the train/test images in `DeepFashion_highres/train` and `DeepFashion_highres/test`.

2. Compute [densepose](https://github.com/facebookresearch/detectron2/tree/master/projects/DensePose):
   1. Install detectron2 following their [installation instructions](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).

   2. Use [apply net](https://github.com/facebookresearch/detectron2/blob/master/projects/DensePose/doc/TOOL_APPLY_NET.md) from [densepose](https://github.com/facebookresearch/detectron2/tree/master/projects/DensePose) and save the train/test results to a pickle file.
   Make sure you download [densepose_rcnn_R_101_FPN_DL_s1x.pkl](https://github.com/facebookresearch/detectron2/blob/master/projects/DensePose/doc/DENSEPOSE_IUV.md#ModelZoo).

   3. Copy `util/pickle2png.py` into your detectron2 DensePose project directory. Using the DensePose environment, convert the pickle file to densepose png images and save the results in `DATASET/densepose` directory, using:
   ```
   python pickle2png.py --pickle_file train_output.pkl --save_path DATASET/densepose/train
   ```

3. Compute [human foreground mask](https://github.com/Engineering-Course/CIHP_PGN). Save results in `silhouette` directory. Or you can download our computed silhouettes for the [training set](https://filebox.ece.vt.edu/~Badour/pose-with-style/data/train_sil.zip) and [testing set](https://filebox.ece.vt.edu/~Badour/pose-with-style/data/test_sil.zip).

4. Compute UV space coordinates:
   1. Compute UV space partial coordinates in the resolution 512x512.
      1. Download the [UV space - 2D look up map](https://filebox.ece.vt.edu/~Badour/pose-with-style/downloads/dp_uv_lookup_256.npy) and save it in `util` folder.
      2. Compute partial coordinates:
      ```
      python util/dp2coor.py --image_file DATASET/DeepFashion_highres/tools/train.lst --dp_path DATASET/densepose/train --save_path DATASET/partial_coordinates/train
      ```

   2. Complete the UV space coordinates offline, for faster training.
      1. Download the pretrained coordinate completion model from [here](https://filebox.ece.vt.edu/~Badour/pose-with-style/downloads/CCM_epoch50.pt).
      2. Complete the partial coordinates.
      ```
      python util/complete_coor.py --dataroot DATASET/DeepFashion_highres --coordinates_path DATASET/partial_coordinates --image_file DATASET/DeepFashion_highres/tools/train.lst --phase train --save_path DATASET/complete_coordinates --pretrained_model /path/to/CCM_epoch50.pt
      ```

5. Download the following in `DATASET/resources`, to apply Face Identity loss:
      1. download the pre-computed [required transformation (T)](https://filebox.ece.vt.edu/~Badour/pose-with-style/downloads/train_face_T.pickle) to align and crop the face.
      2. Download [sphereface net pretrained model](https://filebox.ece.vt.edu/~Badour/pose-with-style/downloads/sphere20a_20171020.pth).

Note: we provide the DeepFashion train/test split of [StylePoseGAN](https://people.mpi-inf.mpg.de/~ksarkar/styleposegan/) [Sarkar et al. 2021]: [train pairs](https://filebox.ece.vt.edu/~Badour/pose-with-style/downloads/sarkar-phase-pairs-train.csv), and [test pairs](https://filebox.ece.vt.edu/~Badour/pose-with-style/downloads/sarkar-phase-pairs-test.csv).
