import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
assert x_train.shape == (50000, 32, 32, 3)
assert x_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)
print("helloworld")


#def generate_model():
#    model = tf.keras.Sequential([
#        # first 
#        tf.keras.layers.Conv2D(32, filters=3,kernel_size=3, activation='relu'),
#        tf.keras.layers.Conv2D(32),
#        tf.keras.layers.MaxPool2D(pool_size=2,strides=2),
#        
#        # second 
#        tf.keras.layers.Conv2D(64, filters=3, kernel_size=3,activation='relu'),
#        tf.keras.layers.MaxPool2D(pool_size=2,strides=2),
#        
#        #fully
#        tf.keras.layers.Flatten(),
#        tf.keras.layers.Dense(1024,activation='relu'),
#        tf.keras.layers.Dense(10,activation='relu')
#    ])
#    return model
#model = generate_model()