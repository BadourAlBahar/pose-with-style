# Pose with Style: Detail-Preserving Pose-Guided Image Synthesis with Conditional StyleGAN
### [[Paper](https://pose-with-style.github.io/asset/paper.pdf)] [[Project Website](https://pose-with-style.github.io/)] [[Output resutls](https://pose-with-style.github.io/results.html)]

Official Pytorch implementation for **Pose with Style: Detail-Preserving Pose-Guided Image Synthesis with Conditional StyleGAN**. Please contact Badour AlBahar (badour@vt.edu) if you have any questions.

<p align='center'>
<img src='https://pose-with-style.github.io/images/teaser.jpg' width='900'/>
</p>


## Requirements
```
conda create -n pws python=3.8
conda activate pws
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```
Intall openCV using `conda install -c conda-forge opencv` or `pip install opencv-python`.
If you would like to use [wandb](https://wandb.ai/site), install it using `pip install wandb`.

## Download pretrained models
You can download the pretrained model [here](https://filebox.ece.vt.edu/~Badour/pose-with-style/pretrained/posewithstyle.pt), and the pretrained coordinate completion model [here](https://filebox.ece.vt.edu/~Badour/pose-with-style/downloads/CCM_epoch50.pt).

Note: we also provide the pretrained model trained on [StylePoseGAN](https://people.mpi-inf.mpg.de/~ksarkar/styleposegan/) [Sarkar et al. 2021] DeepFashion train/test split [here](https://filebox.ece.vt.edu/~Badour/pose-with-style/pretrained/posewithstyle_sarkarsplit.pt). We also provide this split's pretrained coordinate completion model [here](https://filebox.ece.vt.edu/~Badour/pose-with-style/downloads/sarkar_CCM_epoch50.pt).

## Reposing
Download the [UV space - 2D look up map](https://filebox.ece.vt.edu/~Badour/pose-with-style/downloads/dp_uv_lookup_256.npy) and save it in `util` folder.

We provide sample data in `data` directory. The output will be saved in `data/output` directory.
```
python inference.py --input_path ./data --CCM_pretrained_model path/to/CCM_epoch50.pt --pretrained_model path/to/posewithstyle.pt
```

To repose your own images you need to put the input image (input_name+'.png'), dense pose (input_name+'_iuv.png'), and silhouette (input_name+'_sil.png'), as well as the target dense pose (target_name+'_iuv.png') in `data` directory.
```
python inference.py --input_path ./data --input_name fashionWOMENDressesid0000262902_3back --target_name fashionWOMENDressesid0000262902_1front --CCM_pretrained_model path/to/CCM_epoch50.pt --pretrained_model path/to/posewithstyle.pt
```

## Garment transfer
Download the [UV space - 2D look up map](https://filebox.ece.vt.edu/~Badour/pose-with-style/downloads/dp_uv_lookup_256.npy) and the [UV space body part segmentation](https://filebox.ece.vt.edu/~Badour/pose-with-style/downloads/uv_space_parts.npy). Save both in `util` folder.
The UV space body part segmentation will provide a generic segmentation of the human body. Alternatively, you can specify your own mask of the region you want to transfer.

We provide sample data in `data` directory. The output will be saved in `data/output` directory.
```
python garment_transfer.py --input_path ./data --CCM_pretrained_model path/to/CCM_epoch50.pt --pretrained_model path/to/posewithstyle.pt --part upper_body
```

To use your own images you need to put the input image (input_name+'.png'), dense pose (input_name+'_iuv.png'), and silhouette (input_name+'_sil.png'), as well as the garment source target image (target_name+'.png'), dense pose (target_name+'_iuv.png'), and silhouette (target_name+'_sil.png') in `data` directory. You can specify the part to be transferred using `--part` as `upper_body`, `lower_body`, `full_body` or `face`. The output as well as the part transferred (shown in red) will be saved in `data/output` directory.
```
python garment_transfer.py --input_path ./data --input_name fashionWOMENSkirtsid0000177102_1front --target_name fashionWOMENBlouses_Shirtsid0000635004_1front --CCM_pretrained_model path/to/CCM_epoch50.pt --pretrained_model path/to/posewithstyle.pt --part upper_body
```

## DeepFashion Dataset
To train or test, you must download and process the dataset. Please follow instructions in [Dataset and Downloads](https://github.com/BadourAlBahar/pose-with-style/blob/main/DATASET.md).

You should have the following downloaded in your `DATASET` folder:
```
DATASET/DeepFashion_highres
 - train
 - test
 - tools
   - train.lst
   - test.lst
   - fashion-pairs-train.csv
   - fashion-pairs-test.csv

DATASET/densepose
 - train
 - test

DATASET/silhouette
 - train
 - test

DATASET/partial_coordinates
 - train
 - test

DATASET/complete_coordinates
 - train
 - test

DATASET/resources
 - train_face_T.pickle
 - sphere20a_20171020.pth
```


## Training
Step 1: First, train the reposing model by focusing on generating the foreground.
We set the batch size to 1 and train for 50 epochs. This training process takes around 7 days on 8 NVIDIA 2080 Ti GPUs.
```
python -m torch.distributed.launch --nproc_per_node=8 --master_port XXXX train.py --batch 1 /path/to/DATASET --name exp_name_step1 --size 512 --faceloss --epoch 50
```
The checkpoints will be saved in `checkpoint/exp_name`.

Step 2: Then, finetune the model by training on the entire image (only masking the padded boundary).
We set the batch size to 8 and train for 10 epochs. This training process takes less than 2 days on 2 A100 GPUs.
```
python -m torch.distributed.launch --nproc_per_node=2 --master_port XXXX train.py --batch 8 /path/to/DATASET --name exp_name_step2 --size 512 --faceloss --epoch 10 --ckpt /path/to/step1/pretrained/model --finetune
```

## Testing
To test the reposing model and generate the reposing results:
```
python test.py /path/to/DATASET --pretrained_model /path/to/step2/pretrained/model --size 512 --save_path /path/to/save/output
```
Output images will be saved in `--save_path`.

You can find our reposing output images [here](https://pose-with-style.github.io/results.html).

## Evaluation
We follow the same evaluation code as [Global-Flow-Local-Attention](https://github.com/RenYurui/Global-Flow-Local-Attention/blob/master/PERSON_IMAGE_GENERATION.md#evaluation).


## Bibtex
Please consider citing our work if you find it useful for your research:

	@article{albahar2021pose,
	    title   = {Pose with {S}tyle: {D}etail-Preserving Pose-Guided Image Synthesis with Conditional StyleGAN},
      author  = {AlBahar, Badour and Lu, Jingwan and Yang, Jimei and Shu, Zhixin and Shechtman, Eli and Huang, Jia-Bin},
	    journal = {ACM Transactions on Graphics},
      year    = {2021}
	}


## Acknowledgments
This code is heavily borrowed from [Rosinality: StyleGAN 2 in PyTorch](https://github.com/rosinality/stylegan2-pytorch).
