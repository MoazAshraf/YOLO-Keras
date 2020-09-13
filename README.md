# YOLO Implementation in Keras (TensorFlow 2)
In this project, I attempt to implement YOLOv1 as described in the paper [You Only Look Once](https://arxiv.org/pdf/1506.02640.pdf) using TensorFlow 2's Keras API implementation. I use the [yolov1.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov1.cfg) file to generate the model in Keras.

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
- [pjreddie.com/darknet/yolov1/](https://pjreddie.com/darknet/yolov1/)
- [github.com/pjreddie/darknet](https://github.com/pjreddie/darknet)
- [github.com/qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3)
- [Implementing YOLOV1 from scratch using Keras Tensorflow 2.0][https://www.maskaravivek.com/post/yolov1/]
- [How to implement a YOLO (v3) object detector from scratch in PyTorch: Part 2
](https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-2/)
- [How to implement a YOLO (v3) object detector from scratch in PyTorch: Part 3
](https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-3/)

## License
MIT License, check the [LICENSE](LICENSE) file.