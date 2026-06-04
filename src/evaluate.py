from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import load_model

import seaborn as sns
import matplotlib.pyplot as plt

model = load_model(
    "models/best_model.h5"
)

test_gen = ImageDataGenerator(
    rescale=1./255
).flow_from_directory(
    "data/processed/test",
    target_size=(64,64),
    batch_size=32,
    shuffle=False
)

preds = model.predict(test_gen)

y_pred = np.argmax(
    preds,
    axis=1
)

y_true = test_gen.classes

labels = list(
    test_gen.class_indices.keys()
)

print(
    classification_report(
        y_true,
        y_pred,
        target_names=labels
    )
)

cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(figsize=(12,10))

sns.heatmap(
    cm,
    annot=False,
    fmt='d',
    xticklabels=labels,
    yticklabels=labels
)

plt.title("Confusion Matrix")

plt.show()
