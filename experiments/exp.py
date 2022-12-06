import tensorflow as tf
from tensorflow import keras
import numpy as np

imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print("Training entries: {}, labels{}".format(len(train_data), len(train_labels)))
print("Sample Training Data 1:", train_data[0], "\nSample Training Label 1:", train_labels[0])



from tensorflow.keras.preprocessing.text  import Tokenizer
from tensorflow.keras.utils import pad_sequences

sentences = [
    "I love my dog",
    "I love my cat",
    "You love my dog!",
    "Do you think my dog is amazing?"
]

tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)

padded = pad_sequences(sequences, maxlen=10)
padded1 = pad_sequences(sequences, padding='post', maxlen=10)
padded2 = pad_sequences(sequences, maxlen=2)
print("\nSequences = ", sequences)
print("\nPadded Sequences:\n")
print("Default Padding\n", padded)
print("Post Padding\n", padded1)
print("Truncation\n", padded2)

word_index = imdb.get_word_index()
word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 5
print("Length of DATA 1 before Padding:   ", len(train_data[0]), "\nLength of DATA 2 before Padding:   ", len(train_data[1]))

train_data = keras.utils.pad_sequences(train_data, value=word_index["<PAD>"], padding='post', maxlen=256)
test_Data = keras.utils.pad_sequences(test_data, value=word_index["<PAD>"], padding='post', maxlen=256)

print("Length of DATA 1 after Padding:   ", len(train_data[0]), "\nLength of DATA 2 after Padding:   ", len(train_data[1]))


vocab_size = 10000
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

history = model.fit(partial_x_train, partial_y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val))

model.save('model.h5')
print("Test Data 20: ", test_data[20], "\n Test Label 20: ", test_labels[20])
results = model.evaluate(test_data, test_labels)
print(results)


