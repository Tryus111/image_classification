import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from kerastuner.tuners import RandomSearch

# Training set
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

training_set = train_datagen.flow_from_directory(
        '/Users/user/Documents/Files/Project/repo/image_classification/seg_train',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

# Test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
        '/Users/user/Documents/Files/Project/repo/image_classification/seg_test',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

hidden_layers = [300, 350, 400]
output_layers = 6
dropout_rate = [0.3, 0.5, 0.7]

def create_models(x):
    cnn = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters = 32, kernel_size = 4, activation = 'relu', input_shape = [64, 64, 3]),
        tf.keras.layers.MaxPool2D(pool_size = 3, strides = 3),
        tf.keras.layers.Conv2D(filters = 32, kernel_size = 4, activation = 'relu'),
        tf.keras.layers.MaxPool2D(pool_size = 3, strides = 3),
        tf.keras.layers.Flatten()
                                ])
    # Hyperparameters
    x_hidden_layers = x.Choice('hidden_layers', values=hidden_layers)
    x_dropout_rate = x.Choice('dropout_rate', values=dropout_rate)
    
    cnn.add(tf.keras.layers.Dense(units=x_hidden_layers, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    cnn.add(tf.keras.layers.Dropout(x_dropout_rate))
    cnn.add(tf.keras.layers.Dense(output_layers, activation='softmax'))

    cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return cnn

tuner = RandomSearch(
    create_models,
    objective='val_accuracy',
    max_trials=7,
    executions_per_trial=1,
    directory='tuning',
    project_name='image_classification'
)

tuner.search(training_set, epochs=20, validation_data=test_set)

best_model = tuner.get_best_models(num_models=1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

print('Best Hyperparameters: ', best_hyperparameters.values)

