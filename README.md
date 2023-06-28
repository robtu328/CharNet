# Convolutional Character Networks

This project hosts the testing code for CharNet, described in our paper:

    Convolutional Character Networks
    Linjie Xing, Zhi Tian, Weilin Huang, and Matthew R. Scott;
    In: Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2019.

   
## Installation
Package dependecy by using Anaconda
Envirnment build up by using conda commands.

```
conda create --name cda113 python=3.8
conda activate cda113

conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -c conda-forge opencv
conda install -c anaconda pyyaml
conda install -c conda-forge tqdm
conda install -c conda-forge tensorboardX
conda install -c conda-forge editdistance
conda install -c conda-forge anyconfig
conda install -c conda-forge munch
conda install -c anaconda scipy
conda install -c anaconda scikit-learn
conda install -c conda-forge shapely
conda install -c conda-forge pyclipper
conda install -c anaconda gevent
conda install -c anaconda gevent-websocket
conda install -c anaconda flask
conda install -c conda-forge imgaug
conda install -c conda-forge h5py
conda install -c conda-forge gputil
conda install -c conda-forge pympler
conda install -c anaconda cython
conda install -c conda-forge yacs
conda install -c conda-forge psutil

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
2. For ICDAR 2015, please run the following command line. Please replace `images_dir` with the directory containing ICDAR 2015 testing images.
   The results will be in `results_dir`.

Validation

    ```
    python ./tools/valid_net.py configs/icdar2015_hourglass88.yaml ../data/icdar2015/test_images /home/robtu/Github/CharNet/result/.
    ```
    
#    python tools/test_net.py configs/icdar2015_hourglass88.yaml <images_dir> <results_dir>


## Configuration files
1. ./config/icdar2015_hourglass88.yaml
Parameter WEIGHT means the loaded model directory setting. It's useful for the continuouse training from a unfinished traning process.
Parameter CHAR_DICT_FILE is the character mapping table
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
3. ./config/seg_base.yaml
   Configuration for dataset pre/post process
   Strated from 'Experiment' section.
   Please check the remark within the code. 
   ```
    - name: train_data_game
      class: SynthDataset
      data_dir:
        - '/home/robtu/Github/data/games/metadata/'        #directory of dataset image. 
        
      mat_list:
        - '/home/robtu/Github/data/games/gt_game.mat'      #file of dataset information, including bonding box, text line. . 
      processes:                                           #pre-process of the image in the dataset. 
        - class: AugmentDetectionData
          augmenter_args:
              #- ['Fliplr', 0.5]                           #Flip     
              - {'cls': 'Affine', 'rotate': [-5, 5]}       #Rotate
              # Resize each image’s height to 50-75% of its original size and width to either 16px or 32px or 64px:
              # aug = iaa.Resize({"height": (0.5, 0.75), "width": [16, 32, 64]})
              - ['Resize', {'width': 1024, 'height': 512}] #Resize
          only_resize: False
          keep_ratio: False                                #keep image width/height ratio or  not. 
          #keep_ratio: True
        - class: RandomCropData
          size: [1024, 512]                                #Random Crop image. 
          max_tries: 10
        - class: MakeICDARDat
        - class: MakeBorderMap
          charindx: 'datasets/ICDAR2015/test/char_dict.txt'  
          yamlfile: 'datasets/ICDAR2015/test/rect_bbox.yaml'
        - class: NormalizeImage
          norm_type: 'lib'
        - class: FilterKeys
          superfluous: ['polygons', 'filename', 'shape', 'ignore_tags', 'is_training',  'ignore_tags_char', 'ignore_tags_char']
      mode: "train"
      seqList: True                                        #random image index or not. 
      #snumber: 1000                                       #use partial of dataset. 
      #snumber: 10000

   - name: valid_data_game
     class: SynthDataset
     data_dir:
       - '/home/robtu/Github/data/games/metadata/'
     mat_list:
       - '/home/robtu/Github/data/games/gt_game.mat'
     processes:
       - class: AugmentDetectionData
         augmenter_args:
              - ['Resize', {'width': 1024, 'height': 512}]
          only_resize: False
          keep_ratio: False
          #keep_ratio: True
        - class: RandomCropData
          size: [1024, 512]
          max_tries: 10
        - class: MakeICDARData
        - class: MakeBorderMap
          charindx: 'datasets/ICDAR2015/test/char_dict.txt'  
          yamlfile: 'datasets/ICDAR2015/test/rect_bbox.yaml'
        - class: NormalizeImage
          norm_type: 'lib'
        - class: FilterKeys
          superfluous: ['polygons', 'filename', 'shape', 'ignore_tags', 'is_training',  'ignore_tags_char', 'ignore_tags_char'
      mode: "valid"
      seqList: True
      #snumber: 1000
      #snumber: 10000
 
   - name: 'Experiment'
     main:
        train: train_game                            #train_basket, train_game, train_criminal   (3 datasets. )
        valid: valid_game                            #valid_basket, valid_game, valid_criminal   (3 datasets. )
     train_game: 
        class: TrainSettings
        data_loader: 
            class: DataLoader
            dataset: ^train_data_game
            batch_size: 2                            # batch size
            num_workers: 2
        debug_out: False                             # final result cv2.imshow
        debug_class: False                           # word/char class checking cv2.imshow
        debug_box: False                             # word/char bonding box checking cv2.imshow
        epochs: 60
    valid_game: 
        class: TrainSettings
        data_loader: 
            class: DataLoader
            dataset: ^valid_data_game
            batch_size: 2
            num_workers: 2
        debug_out: False
        debug_class: False
        debug_box: False
        epochs: 1       
   ``` 
5. ./charnet/config/default.py
   Configuration for model behavior. Architecture change in DCN needs to be configured here. 
   
   ```
    from yacs.config import CfgNode as CN
    from charnet.modeling.backbone.hourglass import hourglass88, hourglass88GCN


    _C = CN()

    _C.INPUT_SIZE = 2280
    _C.SIZE_DIVISIBILITY = 1
    _C.WEIGHT= ""

    _C.CHAR_DICT_FILE = ""
    _C.WORD_LEXICON_PATH = ""

    #_C.WORD_MIN_SCORE = 0.95
    _C.WORD_MIN_SCORE = 0.9
    _C.WORD_NMS_IOU_THRESH = 0.15
    #_C.CHAR_MIN_SCORE = 0.9
    _C.CHAR_MIN_SCORE = 0.25
    _C.CHAR_NMS_IOU_THRESH = 0.3
    #_C.CHAR_NMS_IOU_THRESH = 0.1
    _C.MAGNITUDE_THRESH = 0.2

    _C.WORD_STRIDE = 4
    _C.CHAR_STRIDE = 4
    _C.NUM_CHAR_CLASSES = 68

    _C.WORD_DETECTOR_DILATION = 1
    _C.RESULTS_SEPARATOR = chr(31)
    #_C.reg_mode = 'cnn'                        #Recognization branch use  Conv2d 
    _C.reg_mode = 'dcn'                         #Recognization branch use  Conv2d 
    _C.backbone_mode = 'hourglass88'            #original hourglass backbone
    #_C.backbone_mode = 'hourglass88GCN'        #replace Conv2d at beginning of hourglass by using DCN
    #_C.word_box_mode = 'box'                   #postprocess valid char class refer to predict word bonding box
    _C.word_box_mode = 'pixel'                  #postprocess valid char class refer to predict word class
    _C.loss_lenbx_eq_lentx_chk = True           #loss calculation check unmatched charbox size and text length exit to sys or not. 
   ``` 
   


## Dataset
Dataset is in matllab format

Example code for read mat filel 
   ```
    import scipy.io


    data_dir='/home/robtu/Github/data/SynthText/SynthText/'
    mat_list='gt.mat'

    print("Load Mat file ", data_dir)
    mat = scipy.io.loadmat(data_dir+mat_list)
    print("Done...")

    image_list = mat['imnames'][0]                        #image path within dataset
    gt_word_list = mat['wordBB'][0]                       #WORD bonding box
    gt_char_list = mat['charBB'][0]                       #CHAR bonding box
    txt_list = mat['txt'][0]                              #CHAR text

    dbLen = len(image_list)

   ``` 
There is a program to  conver xml dataset format to mat format. 
./tool/XML2mat.py # convert xml format to matlab format 
     -i image directory
     -x xml directory
The structure of dataset. 
     Image directory
     ./metadata/image/[sub-dir1]
     ./metadata/image/[sub-dir2]
     ./metadata/image/[sub-dir3]
     XML directory
     ./metadata/XML/[sub-dir1]
     ./metadata/XML/[sub-dir2]
     ./metadata/XML/[sub-dir3]

sub-dir1/2/3 within image direcoty and xml directory needs to be thes same
   ```
    python ./XML2mat.py -i ./metadata -x ./metadata ./    #-i image directory is at ./metadata, -x xml directory is at ./metadata mat file is at ./
   ``` 
There is a program to check mat file data correct or not. 
./tool/matVerify.py # verify mat format legal or not.  
   ```
    python ./matVerify.py  gt_game.mat    #verify mat fille gt_game.mat
   ``` 
Criminal dataset conversion uses XML2mat_Criminal.py and matVerify_Criminal.py
    

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
