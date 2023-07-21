import os
import cv2

from VitPose.inference import ViT

vit = ViT()


img = cv2.imread("img.png")
points, plots, cropBoxes = vit.detectPose(img)
for plot in plots:
    cv2.imshow('', plot)
    cv2.waitKey(0)
'''


video = 'vid2.webm'
cap = cv2.VideoCapture(video)
# Check if camera opened successfully
if (cap.isOpened()== False):
  print("Error opening video stream or file")

  #video VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video.split(".")[0] + "_output.mp4" , fourcc, 60, (3840, 2160))

# Read until video is completed
counter =0
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    # Display the resulting frame
    points, plots, cropBoxes = vit.detectPose(frame)
    for i, plot in enumerate(plots):
        cropBox = cropBoxes[i]

        frame[cropBox[0]:cropBox[1],cropBox[2]:cropBox[3]] = plot
    video.write(frame)
    counter+=1
  else:
    break

video.release()
'''
