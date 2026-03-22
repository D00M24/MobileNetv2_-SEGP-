import tensorflow as tf
import shutil

# 1. Load your Keras 3 model
model = tf.keras.models.load_model('Final_Dog_Posture_Model.keras')

# 2. Export it as a pure TensorFlow SavedModel 
model.export('./saved_model')

# 3. Convert the pure SavedModel into a TFJS Graph Model
!pip install tensorflowjs
!tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model ./saved_model ./web_model

# 4. Zip for download
shutil.make_archive('web_model', 'zip', 'web_model')
print("Download web_model.zip")
