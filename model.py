from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def create_model(input_shape, output_units):
    model = Sequential()

    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.3))

    model.add(LSTM(128))
    model.add(Dense(128))
    model.add(Dense(output_units, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam'
    )

    return model
