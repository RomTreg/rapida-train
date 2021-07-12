import cv2
import numpy as np
import os

directory = os.fsencode("/home/ubuntu/Downloads/set2/SegmentationClass/")
for file in os.listdir(directory):
     filename = os.fsdecode(file)
     if filename.endswith(".jpg") or filename.endswith(".png"): 
         # print(os.path.join(directory, filename))
        image = cv2.imread("/home/ubuntu/Downloads/set2/SegmentationClass/" + filename)
        image_resized = cv2.resize(image, (608, 416))
        gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        
        lower_bolt = np.array([1], dtype = "uint8")
        upper_bolt = np.array([126], dtype = "uint8")
        lower_head = np.array([127], dtype = "uint8")
        upper_head = np.array([255], dtype = "uint8")
        mask_bolt = cv2.inRange(gray, lower_bolt, upper_bolt)
        mask_head = cv2.inRange(gray, lower_head, upper_head)
        finalImage = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
        finalImage[mask_bolt>0] = 1
        finalImage[mask_head>0] = 2
        #cv2.imshow("245", mask_bolt)
        # cv2.imshow("123", finalImage)
        # cv2.imshow("543", image_resized)
        #cv2.waitKey(0)
        cv2.imwrite("/home/ubuntu/dataset_seg/annotations_prepped_test/" + filename, finalImage)