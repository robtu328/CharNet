# Convolutional Character Networks

This project hosts the testing code for CharNet, described in our paper:

    Convolutional Character Networks
    Linjie Xing, Zhi Tian, Weilin Huang, and Matthew R. Scott;
    In: Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2019.

   
## Installation

```
pip install torch torchvision
python setup.py build develop
```

## Install Synthdata

```
Download SynthText dataset from the Link
https://www.robots.ox.ac.uk/~vgg/data/scenetext/

SynthText dataset path is defined in 
configs/seg_base.yaml

I create a SynthText subset mat file. 
Please copy dataset/SynthText/gtsub.mat to SynthText directory

```



## Run
#1. Please run `bash download_weights.sh` to download our trained weights. 

Training
1. For SynthText/Jersey/Criminal please run command file belw. 
 
   ```
   python tools/train_net.py configs/icdar2015_hourglass88.yaml ../data/icdar2015/test_images /home/robtu/Github/CharNet/result/.    
   ``` 
2. For ICDAR 2015, please run the following command line. Please replace `images_dir` with the directory containing ICDAR 2015 testing images. The results will be in `results_dir`.

Validation
python ./tools/valid_net.py configs/icdar2015_hourglass88.yaml ../data/icdar2015/test_images /home/robtu/Github/CharNet/result/.
    ```
    python ./tools/valid_net.py configs/icdar2015_hourglass88.yaml ../data/icdar2015/test_images /home/robtu/Github/CharNet/result/.
    ```
    
#    python tools/test_net.py configs/icdar2015_hourglass88.yaml <images_dir> <results_dir>


## Configuration files
1. ./config/icdar2015_hourglass88.yaml
   ```
INPUT_SIZE: 2280
#WEIGHT: "weights/icdar2015_hourglass88.pth"
#WEIGHT: "weights/model_save.pth"
#WEIGHT: "weights/model_save_dcn.pth"
#WEIGHT: "weights/model_save_dcnReg.pth"
#WEIGHT: ""
CHAR_DICT_FILE: "datasets/ICDAR2015/test/char_dict.txt"
WORD_LEXICON_PATH: "datasets/ICDAR2015/test/GenericVocabulary.txt"
RESULTS_SEPARATOR: ","
SIZE_DIVISIBILITY: 128
       
   ``` 
2. ./config/seg_base.yaml
   ```
       
   ``` 
3. ./charnet/config/default.py
   ```
       
   ``` 
4. 3q424


## Dataset


## Dataset mat file preparation


## Citation

If you find this work useful for your research, please cite as:

    @inproceedings{xing2019charnet,
    title={Convolutional Character Networks},
    author={Xing, Linjie and Tian, Zhi and Huang, Weilin and Scott, Matthew R},
    booktitle={Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
    year={2019}
    }
    
## Contact

For any questions, please feel free to reach: 
```
github@malongtech.com
```


## License

CharNet is CC-BY-NC 4.0 licensed, as found in the [LICENSE](LICENSE) file. It is released for academic research / non-commercial use only. If you wish to use for commercial purposes, please contact sales@malongtech.com.
