# YOLO Implementation in Keras (TensorFlow 2)
In this project, I attempt to implement YOLOv1 as described in the paper [You Only Look Once](https://arxiv.org/pdf/1506.02640.pdf) using TensorFlow 2's Keras API implementation. I use the [yolov1.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov1.cfg) file and the [pretrained model weights](http://pjreddie.com/media/files/yolov1/yolov1.weights) provided by the authors of the paper.

## YOLO Architecture
<div style="text-align: center;">
<img src="yolo-architecture.png" width="928">
</div>

## Dependencies
All the packages used in this project are in the [requirements.txt](requirements.txt) file. For example, you can use pip to install them in a virtual environment like so:

```
pip install -r requirements.txt
```

## References
- [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640.pdf)
- [https://pjreddie.com/darknet/yolov1/](https://pjreddie.com/darknet/yolov1/)
- [github.com/qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3)
- [How to implement a YOLO (v3) object detector from scratch in PyTorch: Part 2
](https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-2/)
- [How to implement a YOLO (v3) object detector from scratch in PyTorch: Part 3
](https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-3/)

## License
MIT License, check the [LICENSE](LICENSE) file.