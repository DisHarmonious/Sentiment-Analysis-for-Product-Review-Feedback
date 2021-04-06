import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import re, os
import numpy as np
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.layers import Embedding

###READ DATA
#purposefully omitted

###CLEAN DATA
def clean_data(phrase):
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    phrase = re.sub(r"<br />", " ", phrase)
    phrase = phrase.lower()
    return phrase

cleaned_data=[]
for i in range(len(all_comments)):
    cleaned_data.append(clean_data(all_comments[i]))

###SHUFFLE SPLIT
#purposefully omitted

###TOKENIZE
#purposefully omitted

###HANDLE EMBEDDINGS
#purposefully omitted

###HANDLE PADDING
#TAKEN CARE OF , through vocab embedding

###BUILD MODEL
int_sequences_input = keras.Input(shape=(None,), dtype="int64")
embedded_sequences = embedding_layer(int_sequences_input)
x = layers.Conv1D(128, 5, activation="relu")(embedded_sequences)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(128, 5, activation="relu")(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(128, 5, activation="relu")(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)
preds = layers.Dense(2, activation="softmax")(x)
model = keras.Model(int_sequences_input, preds)
model.summary()

###TRAIN MODEL
x_train = vectorizer(np.array([[s] for s in train_samples])).numpy()
x_val = vectorizer(np.array([[s] for s in val_samples])).numpy()
y_train = np.array(train_labels)
y_val = np.array(val_labels)

model.compile(loss="sparse_categorical_crossentropy", optimizer="rmsprop", metrics=["acc"])
model.fit(x_train, y_train, batch_size=128, epochs=50, validation_data=(x_val, y_val))

###BUILD END-TO-END MODEL
string_input = keras.Input(shape=(1,), dtype="string")
x = vectorizer(string_input)
preds = model(x)
end_to_end_model = keras.Model(string_input, preds)

end_to_end_model.save('models/CNN_SA_50e')

'''
###SAVE MODEL
end_to_end_model.save('CNN_SA_1')

###PREDICT
prob=end_to_end_model.predict(
    [["what a cool product!"]]
)
print(prob)
'''
