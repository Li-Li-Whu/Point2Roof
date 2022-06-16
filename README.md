# Point2Roof
This is the PyToch implementation of the following manuscript:
> Point2Roof: End-to-end 3D building roof modeling from airborne LiDAR point clouds
>
> Li Li, Nan Song, Fei Sun, Xinyi Liu, Ruisheng Wang, Jian Yao
>
This manuscript has been submitted to ISPRS Journal. The source code and dataet will be publicly available after the acceptance of this manuscript.

## Synthetic dataset

To train and test DeepRoof, we construct a new dataset of roof building 3D reconstruction. 
The dataset can be downloaded from https://github.com/Li-Li-Whu/DeepRoof/tree/main/data. 
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
We have provided the train.txt and test.txt used in our enviroment. Then, your can run train.py and test.py.

## Citation

If you find our work useful for your research, please consider citing our paper.
> Point2Roof: End-to-end 3D building roof modeling from airborne LiDAR point clouds
>
> Li Li, Nan Song, Fei Sun, Xinyi Liu, Ruisheng Wang, Jian Yao

## Contact:
Li Li (li.li@whu.edu.cn)







      