# Point2Roof
This is the PyToch implementation of the following manuscript:
> Point2Roof: End-to-end 3D building roof modeling from airborne LiDAR point clouds
>
> Li Li, Nan Song, Fei Sun, Xinyi Liu, Ruisheng Wang, Jian Yao, Shapsheng Cao
>
This manuscript has been accepted by ISPRS Journal.

## Synthetic and real dataset

To train and test DeepRoof, we construct a new dataset of roof building 3D reconstruction. To further evaluate the performance of the proposed Point2Roof, we also construct a small real dataset. 
The real building point clouds are selected from [RoofN3D](https://github.com/sarthakTUM/roofn3d). 
Now, this dataset only has 500 buildings. We will further expand this real dataset. 

The synthetic and real dataset can be downloaded from [[baiduyun](https://pan.baidu.com/s/1Esbpnp30fWHA1_7eXwcYtQ) :nou3]. 
You can directly unzip the file to obtain all train and test sets. 
For each sample, polygon.obj is the groud truth model, and points.xyz is the roof point clouds. 


 

## Usage
Befor you tain the model, you need to install "pc_util" using "setup.py" using the following command:

```shell script
cd into pc_util
python setup.py install
``` 

Required PyTorch 1.8 or newer. The other dependencies are (maybe incomplete):
- sklearn
- tqdm
- scipy

You can install the missed dependencies according to the compilation errors.

## Train and test
After downloading our dataset and code, you need to prepare your train.txt and test.txt.
We have provided the train.txt and test.txt used in our enviroment. Then, your can run train.py and test.py. checkpoint_epoch_90.pth is our trained model.

## Results

We present 16 roof models reconstructed by [2.5D dual contouring](https://qianyi.info/urban.html), [TopoLAP](http://skyearth.org/LiDARPro/), and the proposed Point2Roof in ./results. 


## Citation

If you find our work useful for your research, please consider citing our paper.
> Point2Roof: End-to-end 3D building roof modeling from airborne LiDAR point clouds
>
> Li Li, Nan Song, Fei Sun, Xinyi Liu, Ruisheng Wang, Jian Yao, Shapsheng Cao

In addition, if you use the real dataset, please also consider citing the following paper:

```shell script
@article{wichmann2019roofn3d,
  title={RoofN3D: A database for 3D building reconstruction with deep learning},
  author={Wichmann, Andreas and Agoub, Amgad and Schmidt, Valentina and Kada, Martin},
  journal={Photogrammetric Engineering \& Remote Sensing},
  volume={85},
  number={6},
  pages={435--443},
  year={2019},
  publisher={American Society for Photogrammetry and Remote Sensing}
}
``` 

## Contact:
Li Li (li.li@whu.edu.cn)







      