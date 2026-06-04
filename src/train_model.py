import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.layers import Dropout, BatchNormalization

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)

import argparse

def build_cnn(input_shape=(64,64,3), num_classes=29):

    model = Sequential([

        Conv2D(
            32,
            (3,3),
            activation='relu',
            input_shape=input_shape
        ),

        BatchNormalization(),
        MaxPooling2D(2,2),

        Conv2D(
            64,
            (3,3),
            activation='relu'
        ),

        BatchNormalization(),
        MaxPooling2D(2,2),

        Conv2D(
            128,
            (3,3),
            activation='relu'
        ),

        MaxPooling2D(2,2),

        Flatten(),

        Dense(
            256,
            activation='relu'
        ),

        Dropout(0.5),

        Dense(
            num_classes,
            activation='softmax'
        )
    ])

    return model

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        default="data/processed"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=64
    )

    parser.add_argument(
        "--out",
        default="models/best_model.h5"
    )

    args = parser.parse_args()

    train_dir = os.path.join(
        args.data_dir,
        "train"
    )

    val_dir = os.path.join(
        args.data_dir,
        "val"
    )

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(
        rescale=1./255
    )

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(64,64),
        batch_size=args.batch,
        class_mode='categorical'
    )

    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=(64,64),
        batch_size=args.batch,
        class_mode='categorical'
    )

    num_classes = len(
        train_gen.class_indices
    )

    model = build_cnn(
        input_shape=(64,64,3),
        num_classes=num_classes
    )

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [

        ModelCheckpoint(
            args.out,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),

        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1
        ),

        EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        callbacks=callbacks
    )

    model.save(args.out)
