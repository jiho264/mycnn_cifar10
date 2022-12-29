def identity_block(X, filters, kernel_size):
    X_shortcut = X

    X = tf.keras.layers.Conv2D(filters, kernel_size, padding='SAME')(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)

    X = tf.keras.layers.Conv2D(filters, kernel_size, padding='SAME')(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)

    X = tf.keras.layers.Conv2D(filters, kernel_size, padding='SAME')(X)
    X = tf.keras.layers.BatchNormalization()(X)

    # Add
    X = tf.keras.layers.Add()([X, X_shortcut])
    X = tf.keras.layers.Activation('relu')(X)

    return X


def convolutional_block(X, filters, kernel_size):
    X_shortcut = X

    X = tf.keras.layers.Conv2D(filters, kernel_size, padding='SAME')(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)

    X = tf.keras.layers.Conv2D(filters, kernel_size, padding='SAME')(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)

    X = tf.keras.layers.Conv2D(filters, kernel_size, padding='SAME')(X)
    X = tf.keras.layers.BatchNormalization()(X)

    X_shortcut = tf.keras.layers.Conv2D(
        filters, kernel_size, padding='SAME')(X_shortcut)
    X_shortcut = tf.keras.layers.BatchNormalization()(X_shortcut)

    # Add
    X = tf.keras.layers.Add()([X, X_shortcut])
    X = tf.keras.layers.Activation('relu')(X)

    return X


def ResNet50CL(input_shape=(32, 32, 3), classes=10):
    X_input = tf.keras.layers.Input(input_shape)
    X = X_input

    X = convolutional_block(X, 64, (3, 3))  # conv
    X = identity_block(X, 64, (3, 3))
    X = tf.keras.layers.MaxPooling2D(2, 2, padding='SAME')(X)

    X = convolutional_block(X, 128, (3, 3))  # 64->128, use conv block
    X = identity_block(X, 128, (3, 3))
    X = tf.keras.layers.MaxPooling2D(2, 2, padding='SAME')(X)

    X = convolutional_block(X, 256, (3, 3))  # 128->256, use conv block
    X = identity_block(X, 256, (3, 3))
    X = tf.keras.layers.MaxPooling2D(2, 2, padding='SAME')(X)

    X = convolutional_block(X, 512, (3, 3))  # 256->512, use conv block
    X = identity_block(X, 512, (3, 3))
    X = tf.keras.layers.MaxPooling2D(2, 2, padding='SAME')(X)

    X = tf.keras.layers.GlobalAveragePooling2D()(X)
    X = tf.keras.layers.Dense(10, activation='softmax')(
        X)  # ouput layer (10 class)

    model = tf.keras.models.Model(inputs=X_input, outputs=X, name="ResNet50CL")

    return model
