
from PyQt5 import QtCore, QtWidgets, uic, QtWidgets
# from PyQt5.QtWidgets import * 
# from PyQt5.QtGui import * 
# from PyQt5.QtCore import * 
from actions import ImageViewer
import sys, os, shutil
import time
import sys
import random
import torch
import numpy as np
import cv2
from coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names
from PIL import Image
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from torchvision.transforms import transforms as transforms
import pandas as pd



# this will help us create a different color for each class


Image.MAX_IMAGE_PIXELS = None
img_data = {}

#---------Load Image----------------------------------------

gui = uic.loadUiType("main.ui")[0]     # load UI file designed in Qt Designer
VALID_FORMAT = ('.BMP', '.GIF', '.JPG', '.JPEG', '.PNG', '.PBM', '.PGM', '.PPM', '.TIFF', '.XBM')  # Image formats supported by Qt

def getImages(folder):
    ''' Get the names and paths of all the images in a directory. '''
    image_list = []
    if os.path.isdir(folder):
        for file in os.listdir(folder):
            if file.upper().endswith(VALID_FORMAT):
                im_path = os.path.join(folder, file)
                image_obj = {'name': file, 'path': im_path }
                image_list.append(image_obj)
    return image_list

#-------------------------------------------------------------------

# Change the mode based on requirements
mode = 1

processed_img_folder = 'processed_images'

if mode==1:
    weight_loc = 'weed_mask_rcnnV1.pt'
    coco_names = ['__background__', 'crop', 'weed']
    image_source = 'images'
    thresh = 0.85
elif mode==2:
    weight_loc = 'weed_detection_mask_100epoch.pt'
    coco_names = ['__background__', 'johnson grass', 'purslane', 'field bindweed']
    image_source = 'images1'
    thresh = 0.40


COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))
#-----------------------MainWindow----------------------------------

class Iwindow(QtWidgets.QMainWindow, gui):

    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.setupUi(self)
        self.cntr, self.numImages = -1, -1  # self.cntr have the info of which image is selected/displayed
        self.image_viewer = ImageViewer(self.qlabel_image)
        self.showMaximized()
        self.setup()
        # self.__connectEvents()
        # self.run() 
    
    #------------Setup-----------------
    def setup(self):
        self.progressBar1.hide()
        self.__connectEvents()
        self.diable_btns()

    def folder_managment(self):
        try:
            shutil.rmtree(processed_img_folder)
            os.mkdir(processed_img_folder)
        except:
            pass

    def run(self):
        self.clear_screen()
        self.folder_managment()
        self.progressBar1.show()
        ival = 0
        self.progressBar1.setValue(ival)
        for i in range(35):
            ival = i
            self.progressBar1.setValue(ival)
            time.sleep(0.03)
        img_loc = self.img_directory(image_source)
        global img_data
        count =0
        imgSize = ival+len(img_loc)
        for iimg_path in img_loc:
            try:
                load_img = self.tree_Segmentation(weight_loc, iimg_path ,thresh)
                self.image_viewer.loadImage(load_img)
                self.set_image_Textdata(load_img)
                count += 1
                print(count)
            except:
                pass

            for i in range(ival, imgSize):
                ival += 1
                self.progressBar1.setValue(i)
                time.sleep(0.03)
                break
        for i in range(ival,101):
            ival = i
            self.progressBar1.setValue(ival)
            time.sleep(0.03)
        self.progressBar1.hide()
        print("Model Process has been Ended!")
        self.After_run()

    #-----------progressBar-----------------




    #----------------------------------------

    def After_run(self):
        self.enable_btn()
        self.selectDir()
        # self.Start_process.setEnabled(False)


    def diable_btns(self):
        self.next_im.setEnabled(False)
        self.prev_im.setEnabled(False)
        self.prev_im.setEnabled(False)
        self.zoom_plus.setEnabled(False)
        self.zoom_minus.setEnabled(False)
        self.reset_zoom.setEnabled(False)
        self.toggle_line.setEnabled(False)
        self.toggle_rect.setEnabled(False)
        self.refresh.setEnabled(False)
        self.undo.setEnabled(False)
        self.redo.setEnabled(False)

    def enable_btn(self):
        self.next_im.setEnabled(True)
        self.prev_im.setEnabled(True)
        self.prev_im.setEnabled(True)
        self.zoom_plus.setEnabled(True)
        self.zoom_minus.setEnabled(True)
        self.reset_zoom.setEnabled(True)
        self.toggle_line.setEnabled(True)
        self.toggle_rect.setEnabled(True)
        self.refresh.setEnabled(True)
        self.undo.setEnabled(True)
        self.redo.setEnabled(True)

    #------------------Handel Buttons---------------------
    def __connectEvents(self):
        try:
            # self.open_folder.clicked.connect(self.selectDir)
            self.next_im.clicked.connect(lambda: self.nextImg())
            self.prev_im.clicked.connect(lambda: self.prevImg())
            self.qlist_images.itemClicked.connect(self.item_click)
            # self.save_im.clicked.connect(self.saveImg)

            self.zoom_plus.clicked.connect(lambda: self.image_viewer.zoomPlus())
            self.zoom_minus.clicked.connect(lambda: self.image_viewer.zoomMinus())
            self.reset_zoom.clicked.connect(lambda: self.image_viewer.resetZoom())

            self.toggle_line.toggled.connect(lambda: self.action_line())
            self.toggle_rect.toggled.connect(lambda: self.action_rect())
            self.Start_process.clicked.connect(lambda: self.run())
            self.refresh.clicked.connect(lambda: self.selectDir())
        except:
            pass

    def set_image_Textdata(self, img_addr):
        name = os.path.split(img_addr)
        # img_dic = dict((key, val) for (key, val) in img_data)
        self.set_comprehensive_data(img_data)
        img_val = img_data[name[1]]
        
        no_instances = len(img_val)
        res = []
        for i in img_val:
            if i not in res:
                res.append(i)
        names =''
        for i in res:
            names += '\n'+i
        info = f"Detected crops: {names} \n\nNumber of Instances: {no_instances}"
        self.img_Data.setText(info)

    def set_comprehensive_data(self, img_dic):
        total_inst = 0
        for i in img_dic.values():
            total_inst += len(i)
        res = []
        for i in img_dic.values():
            # print(i)
            for j in i:
                if j not in res:
                    res.append(j)
        class_size = len(res)
        # print(res)
        names =''
        for i in res:
            names += '\n'+i+' '
        info = f"Crop types: {class_size} \n\nDetected crops: {names} \n\nTotal Instances: {total_inst}"
        self.final_data.setText(info)
    
    def clear_screen(self):
        self.qlist_images.clear()
        self.img_Data.clear()
        self.final_data.clear()
        self.qlabel_image.clear()

    def selectDir(self):
        ''' Select a directory, make list of images in it and display the first image in the list. '''
        # open 'select folder' dialog box
        self.qlist_images.clear()
        self.folder = processed_img_folder
        if not self.folder:
            QtWidgets.QMessageBox.warning(self, 'No Folder Selected', 'Please select a valid Folder')
            return
        
        self.logs = getImages(self.folder)
        self.numImages = len(self.logs)

        # make qitems of the image names
        self.items = [QtWidgets.QListWidgetItem(log['name']) for log in self.logs]
        for item in self.items:
            self.qlist_images.addItem(item)

        # display first image and enable Pan 
        self.cntr = 0
        self.image_viewer.enablePan(True)
        
        self.image_viewer.loadImage(self.logs[self.cntr]['path'])
        self.items[self.cntr].setSelected(True)
        self.set_image_Textdata(self.logs[self.cntr]['path'])
        
        # QtWidgets.QMessageBox.warning(self, 'Sorry', 'No Images! in the destination')
        #self.qlist_images.setItemSelected(self.items[self.cntr], True)

        # enable the next image button on the gui if multiple images are loaded
        if self.numImages > 1:
            self.next_im.setEnabled(True)

    def resizeEvent(self, evt):
        if self.cntr >= 0:
            self.image_viewer.onResize()

    def nextImg(self):
        if self.cntr < self.numImages -1:
            self.cntr += 1
            self.image_viewer.loadImage(self.logs[self.cntr]['path'])
            self.items[self.cntr].setSelected(True)
            self.set_image_Textdata(self.logs[self.cntr]['path'])
            # print(self.logs[self.cntr]['path'])
            #self.qlist_images.setItemSelected(self.items[self.cntr], True)
        else:
            QtWidgets.QMessageBox.warning(self, 'Sorry', 'No more Images!')

    def prevImg(self):
        if self.cntr > 0:
            self.cntr -= 1
            self.image_viewer.loadImage(self.logs[self.cntr]['path'])
            self.items[self.cntr].setSelected(True)
            self.set_image_Textdata(self.logs[self.cntr]['path'])
            #self.qlist_images.setItemSelected(self.items[self.cntr], True)
        else:
            QtWidgets.QMessageBox.warning(self, 'Sorry', 'No previous Image!')

    def item_click(self, item):
        self.cntr = self.items.index(item)
        self.image_viewer.loadImage(self.logs[self.cntr]['path'])
        self.set_image_Textdata(self.logs[self.cntr]['path'])

    def action_line(self):
        if self.toggle_line.isChecked():
            self.qlabel_image.setCursor(QtCore.Qt.CrossCursor)
            self.image_viewer.enablePan(False)

    def action_rect(self):
        if self.toggle_rect.isChecked():
            self.qlabel_image.setCursor(QtCore.Qt.CrossCursor)
            self.image_viewer.enablePan(False)

    def action_move(self):
        if self.toggle_move.isChecked():
            self.qlabel_image.setCursor(QtCore.Qt.OpenHandCursor)
            self.image_viewer.enablePan(True)


    #-------------------------MaskRCNN---------------------------
    def draw_segmentation_map(self, image, masks, boxes, labels):
        alpha = 1
        beta = 0.8 # transparency for the segmentation map
        gamma = 2 # scalar added to each sum
        class_colors = {"crop": (255, 0, 0), "weed": (0, 255, 0), 'johnson grass': (0, 255, 0), 'purslane': (0, 255, 0), 'field bindweed': (255, 0, 0)}
        for i in range(len(masks)):
            red_map = np.zeros_like(masks[i]).astype(np.uint8)
            green_map = np.zeros_like(masks[i]).astype(np.uint8)
            blue_map = np.zeros_like(masks[i]).astype(np.uint8)
            # get the label of the current object and look up the color in the dictionary
            label = labels[i]
            color = class_colors.get(label, (255, 255, 255))
            # apply the color mask to the object
            red_map[masks[i] == 1], green_map[masks[i] == 1], blue_map[masks[i] == 1] = color
            # combine all the masks into a single image
            segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
            #convert the original PIL image into NumPy format
            image = np.array(image)
            # convert from RGN to OpenCV BGR format
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # apply mask on the image
            cv2.addWeighted(image, alpha, segmentation_map, beta, gamma, image)
            # draw the bounding boxes around the objects
            bbcolor = color
            cv2.rectangle(image, boxes[i][0], boxes[i][1], color=bbcolor, 
                        thickness=2)
            # put the label text above the objects
            cv2.putText(image, label, (boxes[i][0][0], boxes[i][0][1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, bbcolor, 
                        thickness=2, lineType=cv2.LINE_AA)
        
        return image




    def get_outputs(self, image, model, threshold):
        with torch.no_grad():
            # forward pass of the image through the modle
            outputs = model(image)
        # print(outputs)
        # get all the scores
        scores = list(outputs[0]['scores'].detach().cpu().numpy())
        # index of those scores which are above a certain threshold
        thresholded_preds_inidices = [scores.index(i) for i in scores if i > threshold]
        thresholded_preds_count = len(thresholded_preds_inidices)
        # get the masks
        masks = (outputs[0]['masks']>0.5).squeeze().detach().cpu().numpy()
        # discard masks for objects which are below threshold
        masks = masks[:thresholded_preds_count]
        # print(outputs)
        # get the bounding boxes, in (x1, y1), (x2, y2) format
        boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]  for i in outputs[0]['boxes'].detach().cpu()]
        # discard bounding boxes below threshold value
        boxes = boxes[:thresholded_preds_count]
        newScores = [i for i in scores if i > threshold]
        # get the classes labels
        labels = []
        for i in range(len(scores)):
            if outputs[0]['scores'][i] > threshold:
                # labels = [coco_names[i] for i in outputs[0]['labels'] if i >threshold]
                labels.append(int(outputs[0]['labels'][i].detach().cpu().numpy()))
        labels = [coco_names[i] for i in labels]
        masks = masks.astype(np.uint8)

        return masks, boxes, labels, newScores

    def tree_Segmentation(self, weight_loc, sorce_loc, conf_thresh):
        threshold = conf_thresh
        image_path = sorce_loc

        # initialize the model
        model = torch.load(weight_loc, map_location=torch.device('cpu'))
        # set the computation device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # load the modle on to the computation device and set to eval mode
        model.to(device).eval()
        # transform to convert the image to tensor
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        

        image = Image.open(image_path).convert('RGB')
        #   display(image)
        # keep a copy of the original image for OpenCV functions and applying masks
        orig_image = image.copy()
        name = os.path.split(image_path)[1]
        # transform the image
        image = transform(image)
        # add a batch dimension
        image = image.unsqueeze(0).to(device)
        masks, boxes, labels, scores = self.get_outputs(image, model, threshold)
        # for i in range(len(labels)):
        #     if labels[i] == 'tree':
        #         labels[i] = 'Pomegranate'
        if len(labels):
            img_data[name] = labels
            result = self.draw_segmentation_map(orig_image, masks, boxes, labels)
            img = Image.fromarray(result, "RGB")                        
            try:
                os.mkdir('processed_images')
            except:
                pass
            save_path = f"processed_images/{name}"
            img.save(save_path)
            return save_path
        # else:
        #     count = count - 1
        #     print('No mask Image -- > deleting count..')
    # else:
    #     print("Image NotFound")
        # print(outPut_dictionary)
    # return outPut_dictionary    

    def img_directory(self, path):
        count = 0
        img_loc = []
        for dirname, dirs, files in os.walk(path):
            for filename in files:
                filename_without_extension, extension = os.path.splitext(filename)
                if extension.upper().endswith(VALID_FORMAT):
                    print('found image')
                    print(filename_without_extension)
                    count = count +1
                    print(f"count = {count}")
                    image_path = os.path.join(dirname, filename)
                    img_loc.append(image_path)

        return img_loc
#-------------------------------------------------------------




#-------------------main--------------------------------

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle(QtWidgets.QStyleFactory.create("Cleanlooks"))
    app.setPalette(QtWidgets.QApplication.style().standardPalette())
    parentWindow = Iwindow(None)
    sys.exit(app.exec_())

if __name__ == "__main__":
    #print __doc__
    main()