# Object detection
## Packages:
- tensorflow
- keras
- numpy
- opencv
- gtts
- pydub
- ffmpeg

## Some result
### ssd_inception_v2_coco model
(based on this repo: https://github.com/tensorflow/models/tree/master/research/object_detection)

Speed: 42ms/frame   Coco mAP: 24 (higher is better)

![ssd](C:\Giang\DL\object_detection\images\ssd.gif)

### faster_rcnn_inception_v2_coco model
(based on this repo: https://github.com/tensorflow/models/tree/master/research/object_detection)

Speed: 58ms/frame   Coco mAP: 28

![rcnn](C:\Giang\DL\object_detection\images\faster_rcnn.gif)

### yolo_v3 model
(based on this repo: https://github.com/qqwweee/keras-yolo3.git)

Speed: 51ms/frame   Coco mAP: 33

![yolo](C:\Giang\DL\object_detection\images\yolo.gif)


## Object detection + voice feedback
(based on this repo: https://github.com/jasonyip184/yolo)

### English voice
[![English](C:\Giang\DL\object_detection\images\y.jpg)](C:\Giang\DL\object_detection\images\english.mp4)

### Japanese voice
[![Japanese](C:\Giang\DL\object_detection\images\y.jpg)](C:\Giang\DL\object_detection\images\japanese.mp4)
