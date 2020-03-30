from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer, text_to_word_sequence
from nltk.corpus import stopwords
import numpy as np

model = load_model('models/100d-model_weights-improvement-72-0.453125.hdf5')
model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
print(model.summary())

tweet = "@MojoInTheMorn these Kevin whatever ads are probably the worst thing to happen to the podcasts. I have to listen to the same ad every 5-10 minutes and the guys voice is annoying"

MAX_SEQUENCE_LENGTH = 20

stop_words = set(stopwords.words('english'))
new_stop_words = set(stop_words)
for s in stop_words:
    new_stop_words.add(s.replace('\'', ''))
    pass

# removing @ from default base filter, to remove that whole word, which might be considered as user or page name
base_filters = '\n\t!"#$%&()*+,-./:;<=>?[\]^_`{|}~ '

word_sequences = []

i = str(tweet)
# uprint(i)

i = i.replace('\'', '')
newlist = [x for x in text_to_word_sequence(i, filters=base_filters, lower=True) if not x.startswith("@")]
filtered_sentence = [w for w in newlist if not w in stop_words]
word_sequences.append(filtered_sentence)

# Tokenizing words to word indices

tokenizer = Tokenizer()
tokenizer.fit_on_texts(word_sequences)
word_indices = tokenizer.texts_to_sequences(word_sequences)
word_index = tokenizer.word_index
print("Tokenized to Word indices as ")
print(np.array(word_indices).shape)

# padding word_indices

x_data = pad_sequences(word_indices, maxlen=MAX_SEQUENCE_LENGTH)
print("After padding data")
print(x_data.shape)

predicted_class = model.predict_classes(x_data)

print(f"Predicted: {predicted_class}")