from tensorflow.keras import datasets, layers, models
import tensorflow as tf


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

validation_images = test_images[:8000]
validation_labels = test_labels[:8000]

test_images = test_images[8000:]
test_labels = test_labels[8000:]

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(10, activation='softmax'))

adam = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=70, batch_size=128, 
    validation_data=(validation_images, validation_labels))

# model.save_weights('./pesos')
# model.load_weights('./pesos')

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print(test_acc)