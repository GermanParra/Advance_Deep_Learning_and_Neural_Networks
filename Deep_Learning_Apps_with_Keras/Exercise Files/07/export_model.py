import pandas as pd
import tensorflow.python.keras.backend as K
from keras.models import Sequential
from keras.layers import *
import tensorflow as tf
from keras.callbacks import TensorBoard

training_data_df = pd.read_csv("sales_data_training_scaled.csv")

X = training_data_df.drop('total_earnings', axis=1).values
Y = training_data_df[['total_earnings']].values

# Define the model
model = Sequential()
model.add(Dense(50, input_dim=9, activation='relu', name='layer_1'))
model.add(Dense(100, activation='relu', name='layer_2'))
model.add(Dense(50, activation='relu', name='layer_3'))
model.add(Dense(1, activation='linear', name='output_layer'))
model.compile(loss='mean_squared_error', optimizer='adam')

# Create a TensorBoard logger
logger = TensorBoard(
    log_dir='logs',
    histogram_freq=5,
    write_graph=True
)

# Train the model
model.fit(
    X,
    Y,
    epochs=50,
    shuffle=True,
    verbose=2,
    callbacks=[logger]
)

# Load the separate test data set
test_data_df = pd.read_csv("sales_data_test_scaled.csv")

X_test = test_data_df.drop('total_earnings', axis=1).values
Y_test = test_data_df[['total_earnings']].values

test_error_rate = model.evaluate(X_test, Y_test, verbose=0)
print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))

model_builder = tf.compat.v1.saved_model.Builder("exported_model") #tf.saved_model.builder.SavedModelBuilder("exported_model")

if tf.executing_eagerly():
    tf.compat.v1.disable_eager_execution()
    sess = tf.compat.v1.Session()

inputs = {
    'input': tf.compat.v1.saved_model.build_tensor_info(model.input) #tf.saved_model.utils.build_tensor_info()
}
outputs = {
    'earnings': tf.compat.v1.saved_model.build_tensor_info(model.output) #tf.saved_model.utils.build_tensor_info()
}

signature_def = tf.compat.v1.saved_model.build_signature_def( #tf.saved_model.signature_def_utils.build_signature_def(
    inputs=inputs,
    outputs=outputs,
    method_name=tf.compat.v1.saved_model.signature_constants.PREDICT_METHOD_NAME  #tf.saved_model.signature_constants.PREDICT_METHOD_NAME
)

model_builder.add_meta_graph_and_variables(
    K.get_session(),
    tags=[tf.compat.v1.saved_model.tag_constants.SERVING], #[tf.saved_model.tag_constants.SERVING],
    signature_def_map={
        tf.compat.v1.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def #tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def
    }
)

model_builder.save()
