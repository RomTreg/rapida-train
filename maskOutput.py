import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import cv2
import numpy as np
from keras_segmentation import predict
from keras_segmentation.models.config import IMAGE_ORDERING
from keras_segmentation.data_utils.data_loader import get_image_array, class_colors
from keras_segmentation.predict import visualize_segmentation
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import json
import re
#picture =  "img"
#maskDir = "output/" + picture + ".jpg"



# import argparse

# parser = argparse.ArgumentParser(description='Lay a mask')
# parser.add_argument("--input", help="input to an image.")
# parser.add_argument("--input_mask", help="input to a mask from segmentation.")
# args = parser.parse_args()





def predictBolts(modelDir, imagesDir, outputDir):
    p = re.compile(r'.*(?=\.)', re.MULTILINE)
    class imageObj:
        def __init__(self, initImageDir, maskedImageDir, numberOfCorrordedBolts, numberOfCorrodedHeadBolts):
            self.initImageDir = initImageDir
            self.maskedImageDir = maskedImageDir
            self.numberOfCorrordedBolts = numberOfCorrordedBolts
            self.numberOfCorrordedHeadBolts = numberOfCorrodedHeadBolts

    raw_data = []
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    #imageDir = "input/" + picture + ".jpg"
    #model = predict.model_from_checkpoint_path("/home/ubuntu/unet_BigSet/")
    model = predict.model_from_checkpoint_path(modelDir)
    output_width = model.output_width
    output_height = model.output_height
    input_width = model.input_width
    input_height = model.input_height
    n_classes = model.n_classes
    print(input_width, input_height)
    directory = imagesDir
    allImages = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(directory):
        for file in f:
            if file.endswith(".jpg"):
                allImages.append(os.path.join(r, file))

    for imageDir in allImages:
        img_init = cv2.imread(imageDir)
        img = cv2.resize(img_init, (input_width, input_height))
        x = get_image_array(img, input_width, input_height,
                            ordering=IMAGE_ORDERING)
        pr = model.predict(np.array([x]))[0]
        pr = pr.reshape((output_height,  output_width, n_classes)).argmax(axis=2)

        segm_img = visualize_segmentation(pr, img, n_classes=n_classes,
                                            colors=class_colors, overlay_img=False,
                                            show_legends=False,
                                            class_names=None,
                                            prediction_width=None,
                                            prediction_height=None)

        mask = segm_img.astype(np.uint8)
        #mask = cv2.imread(maskDir)
        # cv2.imshow("123", mask)
        # cv2.waitKey(0)
        numberOfCorrodedBolts = 0
        numberOfCorrodedHeadBolts = 0 
        grayMask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        lower_bolt = np.array([222], dtype = "uint8")
        upper_bolt = np.array([225], dtype = "uint8")
        lower_head = np.array([215], dtype = "uint8")
        upper_head = np.array([217], dtype = "uint8")
        mask_bolt = cv2.inRange(grayMask, lower_bolt, upper_bolt)
        mask_head = cv2.inRange(grayMask, lower_head, upper_head)
        yellowMaskBolt = cv2.bitwise_and(mask,mask,mask = mask_bolt)
        yellowMaskHead = cv2.bitwise_and(mask,mask,mask = mask_head)
        added_image = cv2.addWeighted(img,1,yellowMaskBolt,0.5,0)
        added_image = cv2.addWeighted(added_image,1,yellowMaskHead,0.5,0)
        cntsBolt = cv2.findContours(mask_bolt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cntsHead = cv2.findContours(mask_head, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        for i in range(len(cntsBolt)):
            if cv2.contourArea(cntsBolt[i]) > 20:
                M = cv2.moments(cntsBolt[i])
                center = (int(M["m10"] / M["m00"]),
                            int(M["m01"] / M["m00"]))
                center_of_text = (center[0]-20, center[1] - 20)
                #cv2.putText(added_image,'corodded bolt',center_of_text, cv2.FONT_HERSHEY_DUPLEX, 0.5,(0,0,0),1)
                numberOfCorrodedBolts +=1
        for i in range(len(cntsHead)):
            if cv2.contourArea(cntsHead[i]) > 20:
                M = cv2.moments(cntsHead[i])
                center = (int(M["m10"] / M["m00"]),
                            int(M["m01"] / M["m00"]))
                center_of_text = (center[0]-20, center[1] - 20)
                #cv2.putText(added_image,'corodded head of bolt',center_of_text, cv2.FONT_HERSHEY_DUPLEX, 0.5,(0,0,0),1)
                numberOfCorrodedHeadBolts+=1
        print("Found " + str(numberOfCorrodedBolts + numberOfCorrodedHeadBolts) + " corroded bolts")
        #cv2.putText(added_image,"Found " + str(numberOfCorrodedBolts + numberOfCorrodedHeadBolts) + " corroded bolts",(10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.7,(255,255,255),1)
        #cv2.imshow("result", added_image)
        imName = p.findall(imageDir)
        cv2.imwrite(imName[0] + "_res.jpg", added_image)
        raw_data.append(imageObj(os.path.abspath(imageDir), os.path.abspath(imageDir), numberOfCorrodedBolts, numberOfCorrodedHeadBolts).__dict__)
        #cv2.imshow("input", img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    if len(raw_data) != 0:
        if os.path.isfile(outputDir + 'stat.json') == False:
            jsonFile = open(outputDir + "stat.json", "x")
        with open(outputDir + 'stat.json', 'w', encoding='utf-8') as f:
            json.dump(raw_data, f, ensure_ascii=False, indent=4)
    else:
        print("Error! Empty input dir")
if __name__ == "__main__":
    predictBolts("/home/ubuntu/unet_BigSet_last/", "/home/ubuntu/Downloads/columns/", "output/")