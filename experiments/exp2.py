from tensorflow import keras

model = keras.models.load_model('model.h5')

model.summary()

print('prediction: ', model.predict([[56, 45, 67]]))