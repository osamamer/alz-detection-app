

import keras.layers

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv3D, BatchNormalization, Activation, GlobalAveragePooling3D,
    Dense, Dropout, Add
)
from tensorflow.keras.models import Model
from keras.layers import multiply
from keras.layers import Lambda
# Configuration


def create_model(input_shape=(128, 128, 128, 1), num_classes=3):
    """Create an improved 3D EfficientNet-B0 like architecture"""

    CONFIG = {
            'input_shape': (128, 128, 128),
            'num_classes': 3,
            'validation_split': 0.2,
            'dropout_rate': 0.2,  # Reduced back to original
            'learning_rate': 0.0002,  # Increased
            'weight_decay': 0.00001,  # Reduced
            'label_smoothing': 0.1,  # Added label smoothing
            'batch_size': 4,  # Increased from 2
            'epochs': 100,  # Increased from 50
        }
    def create_se_block(inputs, filters, reduction_ratio=4):
        """Squeeze-and-Excitation block adapted for 3D"""
        se = GlobalAveragePooling3D()(inputs)
        se = Dense(filters // reduction_ratio, activation='swish')(se)
        se = Dense(filters, activation='sigmoid')(se)
        se = Lambda(lambda x: tf.reshape(x, [-1, 1, 1, 1, filters]))(se)
        return multiply([inputs, se])

    def create_mbconv_block(inputs, filters, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                            expansion_factor=6, se_ratio=4, dropout_rate=0.3):
        """Enhanced MBConv block with Squeeze-and-Excitation"""
        input_filters = inputs.shape[-1]
        expanded_filters = input_filters * expansion_factor

        # Expansion phase
        if expansion_factor != 1:
            x = Conv3D(expanded_filters, (1, 1, 1), padding="same", use_bias=False,
                       kernel_regularizer=tf.keras.regularizers.l2(CONFIG['weight_decay']))(inputs)
            x = BatchNormalization(momentum=0.9)(x)
            x = Activation('swish')(x)
        else:
            x = inputs

        # Depthwise convolution
        x = Conv3D(expanded_filters, kernel_size, strides=strides,
                   padding="same", groups=expanded_filters, use_bias=False,
                   kernel_regularizer=tf.keras.regularizers.l2(CONFIG['weight_decay']))(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Activation('swish')(x)

        # Squeeze-and-Excitation
        x = create_se_block(x, expanded_filters, se_ratio)

        # Projection phase
        x = Conv3D(filters, (1, 1, 1), padding="same", use_bias=False,
                   kernel_regularizer=tf.keras.regularizers.l2(CONFIG['weight_decay']))(x)
        x = BatchNormalization(momentum=0.9)(x)

        # Skip connection
        if strides == (1, 1, 1) and input_filters == filters:
            if dropout_rate > 0:
                x = Dropout(dropout_rate)(x)
            x = Add()([x, inputs])

        return x
    inputs = Input(shape=input_shape)

    # Initial convolution
    x = Conv3D(32, (3, 3, 3), strides=(2, 2, 2), padding='same', use_bias=False,
               kernel_regularizer=tf.keras.regularizers.l2(CONFIG['weight_decay']))(inputs)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('swish')(x)

    # EfficientNet-B0 like structure with increased width
    # MBConv1, 32 channels, 1 layer
    x = create_mbconv_block(x, 16, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                            expansion_factor=1, dropout_rate=CONFIG['dropout_rate'])

    # MBConv6, 16 channels, 2 layers
    for _ in range(2):
        x = create_mbconv_block(x, 24, kernel_size=(3, 3, 3),
                                strides=(2, 2, 2) if _ == 0 else (1, 1, 1),
                                expansion_factor=6, dropout_rate=CONFIG['dropout_rate'])

    # MBConv6, 24 channels, 2 layers
    for _ in range(2):
        x = create_mbconv_block(x, 40, kernel_size=(5, 5, 5),
                                strides=(2, 2, 2) if _ == 0 else (1, 1, 1),
                                expansion_factor=6, dropout_rate=CONFIG['dropout_rate'])

    # Increased width in middle layers
    # MBConv6, 40 channels, 3 layers
    for _ in range(3):
        x = create_mbconv_block(x, 96, kernel_size=(3, 3, 3),
                                strides=(2, 2, 2) if _ == 0 else (1, 1, 1),
                                expansion_factor=6, dropout_rate=CONFIG['dropout_rate'])

    # MBConv6, 80 channels, 3 layers
    for _ in range(3):
        x = create_mbconv_block(x, 128, kernel_size=(5, 5, 5), strides=(1, 1, 1),
                                expansion_factor=6, dropout_rate=CONFIG['dropout_rate'])

    # MBConv6, 112 channels, 4 layers
    for _ in range(4):
        x = create_mbconv_block(x, 224, kernel_size=(5, 5, 5),
                                strides=(2, 2, 2) if _ == 0 else (1, 1, 1),
                                expansion_factor=6, dropout_rate=CONFIG['dropout_rate'])

    # Final layers
    x = Conv3D(1280, (1, 1, 1), padding='same', use_bias=False,
               kernel_regularizer=tf.keras.regularizers.l2(CONFIG['weight_decay']))(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('swish')(x)

    # Global pooling with dropout
    x = GlobalAveragePooling3D()(x)
    x = Dropout(CONFIG['dropout_rate'])(x)

    # Final dense layer with label smoothing
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    # Custom learning rate schedule
    initial_learning_rate = CONFIG['learning_rate']
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate,
        first_decay_steps=20,
        t_mul=2.0,
        m_mul=0.9,
        alpha=0.1
    )

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=CONFIG['weight_decay']
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    return model

# Create and display model
