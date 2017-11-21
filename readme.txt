Sample Android project that uses pre-trained TensorFlow model.

Pre-trained model was generated using this script: https://github.com/kodlan/python_ai/blob/master/mnist_keras/mnist.py
It generates model_output/saved_model.pb model.

In order to import pre-trained model into TensorBoard run the following command:
    python3 import_pb_to_tensorboard.py --model_dir model_output/saved_model.pb --log_dir tmp/

