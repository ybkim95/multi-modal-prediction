import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense, GRU, Concatenate, Input
from tensorflow.keras.models import Model
import random
import pickle

# Dummy Input
acoustic = Input((None, 128))
linguistic = Input((768,))
visual = Input((None, 2048))

# Acoustic GRU
x = GRU(64)(acoustic)

# Linguistic FC
y = Dense(64)(linguistic)

# Visual GRU
z = GRU(64)(visual)

# Concat
concat = Concatenate(axis=-1)([x,y,z])

# Fusion
fusion = Dense(units=192)(concat)

# Turn-changing Prediction
pred = Dense(1, activation='sigmoid')(fusion)

model = Model(inputs=[acoustic, linguistic, visual], outputs=pred)
model.summary()

loss_function = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean()
train_acc = tf.keras.metrics.BinaryAccuracy()


@tf.function
def train_step(input, label):

    # GradientTape for the deriv
    with tf.GradientTape() as tape:
        # pred cal
        predictions = model(input)
        # loss cal
        loss = loss_function(label, predictions)
    
    # grad cal
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # Backpropagation
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # loss, acc update
    train_loss(loss)
    train_acc(label, predictions)


with open('audio_sample.pickle', 'rb') as f1:
    acoustic = pickle.load(f1)

with open('linguistic_sample.pickle', 'rb') as f2:
    linguistic = pickle.load(f2)

with open('visual_sample.pickle', 'rb') as f3:
    visual = pickle.load(f3)


labels= [np.array(random.uniform(0,1), dtype=np.float32).reshape(1,1) for _ in range(10)]

train_ds = [(acoustic[i], linguistic[i], visual[i]) for i in range(len(acoustic))]

for epoch in range(10):
    for input, label in zip(train_ds, labels):
        train_step(input, label)

    template = 'Epoch: {}, Loss: {:.4f}, Acc: {:.6f}'
    print (template.format(epoch+1,
                           train_loss.result().numpy(),
                           train_acc.result().numpy()))