from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras import optimizers
from pathlib import Path
import joblib
import json

# Load data set
x_train = joblib.load("x_train2.dat")
y_train = joblib.load("y_train2.dat")
x_test = joblib.load("x_test2.dat")
y_test = joblib.load("y_test2.dat")

# Create a model and add layers
model = Sequential()

model.add(Flatten(input_shape=x_train.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='sigmoid'))

# Compile the model
model.compile(
    loss="categorical_crossentropy",
    # optimizer="adam",
    optimizer = optimizers.SGD(lr=0.00001, momentum=0.9),
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    x_train,
    y_train,
    epochs=100,
    validation_data=(x_test, y_test),
    shuffle=True
)
with open('history2.json', 'w') as f:
    json.dump(history.history, f)

# Save neural network structure
model_structure = model.to_json()
with open('model_structure2.json', 'w') as f:
    f.write(model_structure)

# f.write_text(model_structure)

# Save neural network's trained weights
model.save_weights("model_weights2.h5")
