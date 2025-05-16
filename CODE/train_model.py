import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Lambda
from keras.optimizers import Adam
from keras_facenet import FaceNet
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from custom_data_gen import YourDataGenerator

# Load pre-trained FaceNet model
facenet = FaceNet()

def embedding_to_label(embeddings, threshold=0.5):
    # Assuming embeddings is a list or array of FaceNet embeddings
    distances = facenet.compute_distances(embeddings)  # Your function to compute distances

    # Convert distances to binary predictions based on the threshold
    predictions = [1 if dist < threshold else 0 for dist in distances]

    return predictions

# Define the Siamese model with normalization
def create_siamese_model():
    input_shape = (160, 160, 3)

    input_anchor = Input(shape=input_shape, name='anchor')
    input_positive = Input(shape=input_shape, name='positive')
    input_negative = Input(shape=input_shape, name='negative')

    output_anchor = facenet.model(input_anchor)
    output_positive = facenet.model(input_positive)
    output_negative = facenet.model(input_negative)

    # Flatten the output
    output_anchor = Flatten()(output_anchor)
    output_positive = Flatten()(output_positive)
    output_negative = Flatten()(output_negative)

    # Add Dense layer with 384 units
    output_anchor = Dense(384, activation='relu')(output_anchor)
    output_positive = Dense(384, activation='relu')(output_positive)
    output_negative = Dense(384, activation='relu')(output_negative)

    # Add a normalization layer using Lambda
    output_anchor = Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))(output_anchor)
    output_positive = Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))(output_positive)
    output_negative = Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))(output_negative)

    # Define the Siamese model
    siamese_model = Model(
        inputs=[input_anchor, input_positive, input_negative],
        outputs=[output_anchor, output_positive, output_negative]
    )

    return siamese_model


# Define triplet loss function
def triplet_loss(alpha=0.2):
    def loss(y_true, y_pred):
        (anchor, positive, negative) = tf.split(y_pred, 3, axis=-1)

        # Compute distances
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)

        # Compute the triplet loss
        basic_loss = pos_dist - neg_dist + alpha
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))

        return loss

    return loss

# Compile the Siamese model
siamese_model = create_siamese_model()
siamese_model.compile(optimizer=Adam(learning_rate=0.001), loss=triplet_loss(), metrics=['accuracy'] )

# Create a data generator for training and validation
from custom_data_gen import YourDataGenerator

train_data_generator = YourDataGenerator(r'C:\Users\Hari Krishna\Documents\GitHub\BCTMinor\datasets\TrainDataset', batch_size=32)
val_data_generator = YourDataGenerator(r'C:\Users\Hari Krishna\Documents\GitHub\BCTMinor\datasets\ValidationDataset', batch_size=32)

# Define callbacks, including ModelCheckpoint to save the best weights
checkpoint = ModelCheckpoint("siamese_model_best.h5", save_best_only=True, monitor='loss', mode='min')

# Train the model using fit
model = siamese_model.fit(
    x=train_data_generator,
    validation_data=val_data_generator,
    epochs=10,
    steps_per_epoch=len(train_data_generator),
    validation_steps=len(val_data_generator),
    callbacks=[checkpoint]
)

# Save the trained Siamese model
siamese_model.save(r'C:\Users\Hari Krishna\Documents\GitHub\BCTMinor\savedModel\siamese_model.keras')
siamese_model.summary()


