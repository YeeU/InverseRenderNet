# InverseRenderNet: Learning single image inverse rendering

***!! Check out our new work InverseRenderNet++ [paper](https://arxiv.org/abs/2102.06591) and [code](https://github.com/YeeU/InverseRenderNet_v2), which improves the inverse rendering results and shadow handling.***

This is the implementation of the paper "InverseRenderNet: Learning single image inverse rendering". The model is implemented in tensorflow.

If you use our code, please cite the following paper:

    @inproceedings{yu19inverserendernet,
        title={InverseRenderNet: Learning single image inverse rendering},
        author={Yu, Ye and Smith, William AP},
        booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        year={2019}
    }

## Evaluation

#### Dependencies
To run our evaluation code, please create your environment based on following dependencies:

    tensorflow 1.12.0
    python 3.6
    skimage
    cv2
    numpy

#### Pretrained model
* Download our pretrained model from: [Link](https://drive.google.com/uc?export=download&id=1VKeByvprmWWXSig-7-fxfXs3KA-HG_-P)
* Unzip the downloaded file 
* Make sure the model files are placed in a folder named "irn_model"


#### Test on demo image
You can perform inverse rendering on random RGB image by our pretrained model. To run the demo code, you need to specify the path to pretrained model, path to RGB image and corresponding mask which masked out sky in the image. The mask can be generated by PSPNet, which you can find on https://github.com/hszhao/PSPNet. Finally inverse rendering results will be saved to the output folder named by your argument.

```bash
python3 test_demo.py --model /PATH/TO/irn_model --image demo.jpg --mask demo_mask.jpg --output test_results
```


#### Test on IIW
* IIW dataset should be downloaded firstly from http://opensurfaces.cs.cornell.edu/publications/intrinsic/#download 

* Run testing code where you need to specify the path to model and IIW data:
```bash
python3 test_iiw.py --model /PATH/TO/irn_model --iiw /PATH/TO/iiw-dataset
```

## Training

#### Train from scratch
The training for InverseRenderNet contains two stages: pre-train and self-train.
* To begin with pre-train stage, you need to use training command specifying option `-m` to `pre-train`. 
* After finishing pre-train stage, you can run self-train by specifying option `-m` to `self-train`. 

In addition, you can control the size of batch in training, and the path to training data should be specified.

An example for training command:
```bash
python3 train.py -n 2 -p Data -m pre-train
```

#### Data for training
To directly use our code for training, you need to pre-process the training data to match the data format as shown in examples in `Data` folder. 

In particular, we pre-process the data before training, such that five images with great overlaps are bundled up into one mini-batch, and images are resized and cropped to a shape of 200 * 200 pixels. Along with input images associated depth maps, camera parameters, sky masks and normal maps are stored in the same mini-batch. For efficiency, every mini-batch containing all training elements for 5 involved images are saved as a pickle file. While training the data feeding thread directly load each mini-batch from corresponding pickle file.





