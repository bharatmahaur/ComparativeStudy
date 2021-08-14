# RetinaNet-Tensorflow

## If you want to use this project, you should first create some folders.<br>
checkpoint: to save your weight;<br>
log/train: to write the loss and learning rate while training;<br>
log/val: to write the precision and recall while evaluating;<br>
tfrecords: the training, val dataset;<br>

## Running
The weight files are [here](https://pan.baidu.com/s/19KiLKS77gwPdW9QQgDPpjg), the passwoard is tib7, download it and put it into the checkpoint folder.<br>
Attention: The Restore path in the `test_images.py`, you can change it by yourself.<br>
Finally, run the test_images like this: `python test_images.py --input_image=.... --Single_test=True --Output_dir=...`

## Prepare your own data
First, you should prepare your image and annotation using the form of VOC: jpeg and xml<br>
Then run the `convert_to_tf.py`, pay attention to the path of your images and annotation, you should use your own path<br>
Finally, put your .tfrecords files into tfrecords folder

## Training
Run the `training.py`, and use your own path to save the .ckpt 

## Evaluate
Run the `evaluation.py`

## Attention 
This model doesn't use the pre-trained weight, so the result is not so good. And I will update the code (pre-trained edition) in the future.
