import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import os, sys
import imghdr
import argparse
import glob
import numpy as np
from shapely.geometry import Polygon
from scipy.io import savemat, loadmat
import cv2



#import cv2 as cv
#import numpy as np

ColorTable={'Yellow':(0, 255, 255), "Blue":(255, 0, 0), 'Green':(0, 255, 0), 'Red':(0, 0, 255)}

def draw_polys(bimg, polys, color='Yellow'):

    h=bimg.shape[0]
    w=bimg.shape[1]
    
    #b= np.zeros((8), dtype=np.int32)
    for i in range(len(polys)):
        pts=polys[i].astype('int32')
        cv2.polylines(bimg, [pts], True, ColorTable[color])

    return bimg



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

yellow=(0, 255,255)
red=(0, 0,255)

def drawBoxes(img, boxes, color):
    img_word_ins = img.copy()
    for i in range(len(boxes)):
        pts=boxes[i][:8].reshape((-1, 2)).astype('int')
        cv2.polylines(img_word_ins, [pts], True, (0, 255,255))
    return img_word_ins
        
def vis(img, wordbox, wordtxt):
    img_word_ins = img.copy()
    
    for wb, txt in zip(wordbox, wordtxt): 
        
        cv2.polylines(img_word_ins, [wb.astype(np.int32)],
                      True, (0, 255, 0), 2)
        cv2.putText(
            img_word_ins,
            '{}'.format(txt),
            (wb[0][0].astype('int32'), wb[0][1].astype('int32')), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )
    return img_word_ins

def verify_mat(mat, index=None, eye_check=False):

    imgnames=mat['imnames'][0]
    wordBB=mat['wordBB'][0]
    charBB=mat['charBB'][0]
    texts=mat['txt'][0]
    #eye_check=False
    ellist = []
    
    db_length=len(imgnames)
    for index in range(db_length):
        name=imgnames[index][0]
        wBB = wordBB[index]
        cBB = charBB[index]
        txt = np.char.strip(texts[index])
        
        print("Index=", index, ", Image name = ", name)
    #    if index == 10000:
    #        print ('wBB', wBB)
    #        print ('cBB', cBB)
    #        print ('txt', txt) 
        if (len(wBB) ==0 or len(cBB)==0 or len(txt)==0):
            print("Problem Index=", index, ", Image name = ", name, 'txt len=', len(''.join(txt))) 
            ellist.append(index)
        
        elif(cBB.shape[2] !=len(''.join(txt))):
            print('txt =', txt)
            print("len CBB(",cBB.shape," is not equal len txt(", len(''.join(txt)), ")")
            #ellist.append(index)
        elif(eye_check== True):
            
            
            print('txt = ',  txt)

            #wBB = wBB.reshape(-1, 4, 1)
            #cBB = cBB.reshape(-1, 4, 1)
            txt = preprocess_words(txt)
            cxc = ''.join((''.join(np.reshape(txt, (1, -1)).tolist()[0])).split())

            wBB = np.transpose(wBB, (2,1,0))
            cBB = np.transpose(cBB, (2,1,0))

            img = cv2.imread('./metadata/'+name)

        
            img_box = drawBoxes(img, wBB, red)
            img_box = drawBoxes(img_box, cBB, yellow)
            img_box=vis(img_box, wBB, txt)
            cv2.destroyAllWindows()
            cv2.imshow('gtbox', img_box)
            cv2.waitKey(4000)
        else:
            
            #print("Index=", index, ", Image name = ", name)
            print('txt= ',  txt)
            txt = preprocess_words(txt)
            cxc = ''.join((''.join(np.reshape(txt, (1, -1)).tolist()[0])).split())

            wBB = np.transpose(wBB, (2,1,0))
            cBB = np.transpose(cBB, (2,1,0))
            
            txt_len=len(cBB)
            
            for cidx in range(txt_len):
                ratio_str=""
                for cidx_ref in range (txt_len):
                    if cidx != cidx_ref:
                        cbox1=cBB[cidx]
                        cbox2=cBB[cidx_ref]
                        pcbox1=Polygon(cbox1)
                        pcbox2=Polygon(cbox2)
                        ratio=pcbox1.intersection(pcbox2).area/pcbox1.area
                    else:
                        ratio=1.0
                    
                    ratio_str=ratio_str+" "+ str(round(ratio, 3))
                 
                #print("cidx ", cidx, " ,ratio_str = ", ratio_str)
                 
                    

#                   image_path = [self.data_dir[i]+timg[0] for timg in image_list]
#                for timg, ttxt in zip(gt_word_list, txt_list):
#                   if(len(timg.shape)==2):
#                     timg=timg.reshape(-1, 4, 1)
#                   ttxt=preprocess_words(ttxt)
#                   gt_map.append({'poly':np.transpose(timg, (2,1,0)),'txt':ttxt})
#
#                for timg, ttxt in zip(gt_char_list, txt_list):
#                   if(len(timg.shape)==2):
#                     timg=timg.reshape(-1, 4, 1)
#                   ttxt=''.join((''.join(np.reshape(ttxt, (1, -1)).tolist()[0])).split())
#                   gt_map_char.append({'poly':np.transpose(timg, (2,1,0)),'txt':ttxt})
#
#
    return ellist

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
                      


def update_mat(mat, ellist):


    imgnames=mat['imnames']
    wordBB=mat['wordBB']
    charBB=mat['charBB']
    texts=mat['txt']

    len_before=len(texts[0])
    ellist.reverse()
    for element in ellist:
        print ("delete elemen = ", element)
        imgnames=np.delete(imgnames, element, 1)
        wordBB=np.delete(wordBB, element, 1)
        charBB=np.delete(charBB, element, 1)
        texts=np.delete(texts, element, 1)


    if len_before == len(texts[0]):
        print ("Size is the same, no need to resave it again, len =", len_before)
    
    elif len(texts[0]) == len(wordBB[0]) and len(texts[0]) == len(charBB[0]) and len(texts[0]) == len(imgnames[0]):
        print("Original size =", len_before, "save mat size correct = ", len(texts[0]))
        header=b'MATLAB 5.0 MAT-file Platform: posix, Created on: Sat Apr 23 19:49:17 2023'
        version='1.0'
        globalSet=""
        new_mat={'__header__':header, '__version__': version, '__globals__':globalSet,  \
                 'imnames':imgnames, 'wordBB':wordBB, 'charBB':charBB, 'txt':texts}
        savemat("new"+args.datapath, new_mat)
#drawBBox(args.imgPath, args.xmlPath, args.datapath)
    else:
        print ("All size are not matched ", texts[0]," ", len(wordBB[0])," ", len(charBB[0])," ", len(imgnames[0]))
     

def show_box(mat, pic_name, show=False):
    print("Search PIC name", pic_name)
    
    imgnames=mat['imnames']
    wordBB=mat['wordBB']
    charBB=mat['charBB']
    texts=mat['txt']
    
    index = [idx for idx, s in enumerate(mat['imnames'][0]) if pic_name in s[0]]
    print("This index of the filename is : ", index)
    
    db_name="./metadata/"+ mat['imnames'][0][index][0][0]
    db_wordBB=wordBB[0][index][0].transpose(2,1,0)
    db_charBB=charBB[0][index][0].transpose(2,1,0)
    db_text=texts[0][index][0][0]
    
    if show:
        im = cv2.imread(db_name)
        im_word=draw_polys(im, db_wordBB, color='Yellow')
        cv2.imshow("word Box", im)
        cv2.waitKey()
  
        im_char=draw_polys(im, db_charBB, color='Yellow')
        cv2.imshow("word Box", im)
        cv2.waitKey()
    
    

path = "./"
pic_name =''

parser = argparse.ArgumentParser()
# mat PATH
parser.add_argument('datapath', nargs='?', type=str, default=path, help='Destination Path')
parser.add_argument('-i', '--imgName', type=str, default=pic_name, help='Image Path')


args = parser.parse_args()
print(args)
print(args.datapath)
print(args.imgName)


mat = loadmat(args.datapath)  

if (args.imgName !=''):
    pic_name ='fortest5 495.jpg'
    pic_name = args.imgName
    index=show_box(mat, pic_name)
else:
    ellist=verify_mat(mat, eye_check=False)
    print(ellist)

    if len(ellist) !=0:
        print("Issue data found, update mat")
        update_mat=update_mat(mat, ellist)


 



