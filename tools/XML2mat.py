import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import os, sys
import imghdr
import argparse
import glob
import numpy as np
from shapely.geometry import Polygon
from scipy.io import savemat


#import cv2 as cv
#import numpy as np





class XMLObject:
  def __init__ (self, Ent, img):
    bndbox=Ent.find('bndbox')
    self.startx=int(bndbox.find('xmin').text)
    self.starty=int(bndbox.find('ymin').text)
    self.endx=int(bndbox.find('xmax').text)
    self.endy=int(bndbox.find('ymax').text) 
    self.name = Ent.find('name').text
    self.pose = Ent.find('pose').text
    self.img = img
  def crop(self):
    return self.img.crop((self.startx, self.starty, self.endx, self.endy))

class PersonInst:
  def isInside(self, wordbox, charbox):    
      ratio=charbox.intersection(wordbox).area/charbox.area
      if  ratio > 0.8:
          return True
      else: 
          return False
      
  def __init__ (self, wordbox, charbox):
    
      a= np.zeros((8), dtype=np.int32)
      #top left
      a[0] = wordbox.startx
      a[1] = wordbox.starty
      #top right
      a[2] = wordbox.endx
      a[3] = wordbox.starty
      #bottom right
      a[4] = wordbox.endx
      a[5] = wordbox.endy
      #bottom left
      a[6] = wordbox.startx
      a[7] = wordbox.endy
      wpts=np.array([[a[0], a[1]], [a[2], a[3]], [a[4], a[5]], [a[6], a[7]]], np.int32)
      self.wordbox=np.array([[a[0], a[2], a[4], a[6]], [a[1], a[3], a[5], a[7]]], np.float32)
      self.charbox=[]
      self.text=""
      for char_idx in charbox:
          if char_idx.name != 'text' and char_idx.name !='person':  
              b= np.zeros((8), dtype=np.int32)
              #top Left
              b[0] = char_idx.startx
              b[1] = char_idx.starty
              #top right
              b[2] = char_idx.endx
              b[3] = char_idx.starty
              #bottom right
              b[4] = char_idx.endx
              b[5] = char_idx.endy
              #bottom left
              b[6] = char_idx.startx
              b[7] = char_idx.endy              
              pts=np.array([[b[0], b[1]], [b[2], b[3]], [b[4], b[5]], [b[6], b[7]]], np.int32)             
                  
              if self.isInside(Polygon(wpts), Polygon(pts)):
                  if self.charbox==[] or b[0]> self.charbox[0][0][0]:
                      self.text=self.text+char_idx.name
                      #self.charbox=np.append(self.charbox, [[b[0], b[2], b[4], b[6]], [b[1], b[3], b[5], b[7]]])
                      self.charbox.append(np.array([[b[0], b[2], b[4], b[6]], [b[1], b[3], b[5], b[7]]], np.float32))
                  else:
                      self.text=char_idx.name+self.text
                      #self.charbox=np.append(self.charbox, [[b[0], b[2], b[4], b[6]], [b[1], b[3], b[5], b[7]]])
                      self.charbox.insert(0, np.array([[b[0], b[2], b[4], b[6]], [b[1], b[3], b[5], b[7]]], np.float32))  
      if (self.text==""):
        print("text equal null = ", self.text)                  
  
class XMLImage:
  def __init__ (self, imgEnt, img, imgpath, savePath=""):
    self.imgName=imgEnt.find('filename').text
    self.imgPath=imgpath
    self.img= img  
    self.savePath=savePath

    imgSource=imgEnt.find('source')
    self.dBaseName=imgSource.find('database').text
    
    imgSize=imgEnt.find('size')
    self.width=int(imgSize.find('width').text)
    self.height=int(imgSize.find('height').text)
    self.depth=int(imgSize.find('depth').text)
    
    self.xmlObjList=imgEnt.findall('object')
    self.objList=[]
    self.personList=[]
    self.charBB=[]
    self.wordBB=[]
    self.text=[]
    
    
    for obj in self.xmlObjList:
       self.objList.append(XMLObject(obj, img))
    
    for obj in self.objList:
        if (obj.name == 'person'):
            Persion=PersonInst(obj, self.objList)
            #print("wb = ", Persion.wordbox, ", cb = ", Persion.charbox,", txt =" ,Persion.text)
            self.charBB.extend(Persion.charbox)
            self.wordBB.append(Persion.wordbox)
            self.text.append(Persion.text)
        #print("(",obj.startx, obj.starty, obj.endx, obj.endy, obj.name, ")", obj.endx-obj.startx, obj.endy - obj.starty, round((obj.endy - obj.starty)/(obj.endx-obj.startx),2))
    
    self.charBB=np.array(self.charBB)
    self.wordBB=np.array(self.wordBB)
    self.text=np.array(self.text)
    if(self.charBB.size >0 and self.wordBB.size > 0):
        self.charBB=self.charBB.transpose(1,2,0)
        self.wordBB=self.wordBB.transpose(1,2,0)
    else:
        print("Empty charBB or wordBB")

  def saveObjectImage (self, savelist=[]):
     saveName=self.imgName.replace(' ', '_')
     f, e = os.path.splitext(saveName)
     #print(f, e)
     for obj in self.objList:
#      print(f+'_'+obj.name+e, '       ', obj.name)
      cropImg=obj.crop()
      cropImg.save(self.savePath+f+'_'+obj.name+e)
      #cropImg.show()

  def drawBBox (self, color=(0,0,255), lWidth=3):
     newImg = self.img
     draw = ImageDraw.Draw(newImg)
     
     for obj in self.objList:
       draw.rectangle([(obj.startx, obj.starty),(obj.endx,obj.endy)], fill=None, outline="red", width=lWidth)

     return newImg
    


def retrieve(imgPath, xmlPath, datapath):
    xmldirs = os.listdir( xmlPath )
    imgdirs = os.listdir( imgPath )

    #xmldirs = glob.glob(xmlPath+'/*.xml')
    #imgdirs = glob.glob(imgPath+'/*.jpg')
    
    imgnames=[]
    texts=[]
    charBB=[]
    wordBB=[]
    xmlfiles=[]
    
    for item in xmldirs:
        if os.path.isdir(xmlPath+"/"+item) and os.path.isdir(imgPath+"/"+item):
            print ("Directory: ", xmlPath, "/", item)
            #files = os.listdir(xmlPath+item)
            xmlfiles=np.append(xmlfiles, glob.glob(xmlPath+"/"+item+"/*.xml"))
            #imgnames.append(glob.glob(imgPath+item+"/*.jpg"))
        elif os.path.isfile(xmlPath+item):
            print ("File ", xmlPath+item, " is excluded.")
        else:
            print ("Dir ", imgPath+item , ' is not existed')
    
    for item in xmlfiles:
        if os.path.isfile(item):
            imgname=item.replace(xmlPath, imgPath)
            f, e = os.path.splitext(imgname)
            imgname=f+".jpg"
            if os.path.isfile(imgname):
              xmlfile=ET.parse(item)             
              img = Image.open(imgname)
              imgname=imgname.replace(imgPath+"/", '')
              print ("File Name: ", imgname)
              xmlImage=XMLImage(xmlfile, img, imgname , datapath)
              imgnames=imgnames + [np.str_(imgname)]
              charBB=charBB + [np.array(xmlImage.charBB)]
              wordBB=wordBB + [np.array(xmlImage.wordBB)]
              texts= texts + [np.array(xmlImage.text)]
              #xmlImage.saveObjectImage()
            else:
              print('Image file ', imgname, ' is not existed. ')
              
    #imgnames=np.array(imgnames, dtype='object').reshape([1,-1,1]) 
    imgnames=np.array(imgnames, dtype='object').reshape([1,-1]) 
    charBB=np.expand_dims(np.array(charBB, dtype='object'), axis=0)
    wordBB=np.expand_dims(np.array(wordBB, dtype='object'), axis=0)
    texts=np.expand_dims(np.array(texts, dtype='object'), axis=0)
    
    
    header=b'MATLAB 5.0 MAT-file Platform: posix, Created on: Sat Apr 23 19:49:17 2023'
    version='1.0'
    globalSet=""
    
    mat={'__header__':header, '__version__': version, '__globals__':globalSet,  \
         'imnames':imgnames, 'wordBB':wordBB, 'charBB':charBB, 'txt':texts}
    return mat


def preprocess_words(word_ar):
    words = []
    #print("word_ar = ", word_ar)
    for ii in range(np.shape(word_ar)[0]):
        s = word_ar[ii]
        start = 0
        while s[start] == ' ' or s[start] == '\n':
            start += 1
        for i in range(start + 1, len(s) + 1):
            if i == len(s) or s[i] == '\n' or s[i] == ' ':
                if start != i:
                    words.append(s[start : i])
                start = i + 1
    return words




def verify_mat(mat, index=None):

    imgnames=mat['imgnames'][0]
    charBB=mat['wordBB'][0]
    wordBB=mat['charBB'][0]
    texts=mat['txt'][0]

    db_length=len(imgnames)
    for index in range(db_length):
        wBB = wordBB[index]
        cBB = charBB[index]
        txt = texts[index]

        wBB = wBB.reshape(-1, 4, 1)
        cBB = cBB.reshape(-1, 4, 1)
        txt = preprocess_words(txt)
        cxc = txt=''.join((''.join(np.reshape(txt, (1, -1)).tolist()[0])).split())


        wBB = np.transpose(wBB, (2,1,0))
        cBB = np.transpose(cBB, (2,1,0))






def drawBBox(imgPath, xmlPath, datapath):
    xmldirs = os.listdir( xmlPath )
    imgdirs = os.listdir( imgPath )
    
    for item in xmldirs:
        if os.path.isfile(xmlPath+item):
            f, e = os.path.splitext(item)
            if (e=='.xml'):
              xmlfile=ET.parse(xmlPath+item)
              imgname=glob.glob(imgPath+f+".*")
              for imgItem in imgname:
#                  if(imghdr.what(imgItem)!=None):
                img = Image.open(imgItem)
                xmlImage=XMLImage(xmlfile, img, xmlPath, imgPath, datapath)
                      
                newImg=xmlImage.drawBBox()
                saveName= xmlImage.imgName.replace(' ', '_')
                f, e = os.path.splitext(saveName)
                newImg.save(xmlImage.savePath+f+'_bbx'+e)
                      



      



path = "./"

parser = argparse.ArgumentParser()
# PATH
parser.add_argument('datapath', nargs='?', type=str, default=path, help='Destination Path')
# size Height
parser.add_argument('-i', '--imgPath', type=str, default=path, help='Image Path')
# size Width
parser.add_argument('-x', '--xmlPath', type=str, default=path, help='XML Path')

args = parser.parse_args()
print(args)
print(args.imgPath, args.xmlPath, args.datapath)



         
mat=retrieve(args.imgPath, args.xmlPath, args.datapath)
savemat("gt_basket.mat", mat)
#verify(mat)
#drawBBox(args.imgPath, args.xmlPath, args.datapath)


#img=0
#xmlfile=ET.parse("../2v2Label/IMG_6098 003.xml")
#img = Image.open("../2v2Image/IMG_6098 003.jpg")
#xmlImage=XMLImage(xmlfile, img, "./", "../2v2Image/", "../2v2Image/")
#xmlImage.saveObjectImage()


 



