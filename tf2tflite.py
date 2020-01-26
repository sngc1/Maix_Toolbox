"""
    tf2tflite.py
    ~~~~~~~~~~~~

    Converts TensorFlow2 model file to TensorFlow lite model file.
    Either Keras model file (.h5) or TF2 SavedModel directory can be converted.

    usage:
        python3 tf2tflite.py my_model.h5 my_model.tflite

        python3 tf2tflite.py models/my_model_saved_model my_model.tflite
"""

import sys
import tensorflow as tf

model_path = sys.argv[1]  # either h5 or SavedModel
outfile = sys.argv[2]  # xxxx.tflite

keras_model = (model_path[-3:] == '.h5')

if keras_model:
    model = tf.keras.models.load_model(model_path) 
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
else:
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)

tflite_model = converter.convert()
open(outfile, "wb").write(tflite_model)