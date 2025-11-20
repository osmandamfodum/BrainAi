import tensorflow as tf

model = tf.keras.models.load_model("C:/Users/Osman Al-Hussein/Documents/GitHub/BrainAi/src/Model2.h5", compile=False)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)
