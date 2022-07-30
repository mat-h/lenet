from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Lambda
from keras.backend import argmax, cast
from keras.utils import to_categorical

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential([
    Conv2D(16, (5,5), strides=(4,4), activation='tanh', input_shape=(28,28,1)),    # Layer H1
    Flatten(),
    Dense(12 * 16),  # Layer H2
    Dense(30),  # Layer H3
    Dense(10, activation='softmax') # Output Layer
])

model.summary()

model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=1)

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=16)
