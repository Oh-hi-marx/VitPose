import os
import cv2
from PIL import Image
from ultralytics import YOLO

class yolov8:
    def __init__(self, model = "yolov8m.pt", filter = [], conf= 0.75):
        self.model = YOLO(model)
        self.filter= filter
        self.conf = conf
    def pred(self,img):
        results = self.model.predict(source=img)
        for result in results: #assume 1 image input
            boxes = result.boxes.cpu().numpy()

        finalboxes=[]
        for box in boxes:
            if(box.conf > self.conf):
                if(len(self.filter)>0):
                    for f in self.filter:
                        if(box.cls==f):
                            finalboxes.append(box)
                else:
                    finalboxes.append(box)
        return finalboxes




    def crop(self, img, box, padding = 0 ):
        box = box.xyxy[0]
        h,w,c = img.shape
        #apply padding if not out of bounds
        padding = int((box[2]-box[0])* padding)
        if(box[0]-padding>0):
            box[0] -= padding
        if(box[2]+padding< w):
            box[2] += padding
        if(box[1]-padding >0):
            box[1]-=padding
        if(box[3]+padding< h):
            box[3]+=padding

        crop = img[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
        cropBox = [int(box[1]),int(box[3]),int(box[0]),int(box[2])]
        return [crop, cropBox]
    def cropall(self, img, boxes, padding = 0):
        crops = []
        for box in boxes:
            crops.append(self.crop(img, box, padding))
        return crops


if __name__ == "__main__":
    img = cv2.imread("testimgs/a.jpg")
    img = cv2.imread("testimgs/b.png")
    img = cv2.imread("testimgs/c.png")
    img = cv2.imread("testimgs/e.png")
    yolo = yolov8()
    boxes = yolo.pred(img)
    crops = yolo.cropall(img,boxes)



    for crop in crops:

        cv2.imshow("", crop)
        cv2.waitKey(0)
