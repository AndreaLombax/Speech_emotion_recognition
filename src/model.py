from tensorflow.keras.layers import Dense, Dropout, Activation, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import RMSprop

def get_model(shape, loss='categorical_crossentropy', optimizer=RMSprop(lr=0.00001, decay=1e-6), metrics=['accuracy']):
    model = Sequential()
    model.add(Conv1D(256, 8, padding='same',input_shape=(shape,1)))  # X_train.shape[1] = No. of Columns
    model.add(Activation('relu'))

    model.add(Conv1D(256, 8, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(MaxPooling1D(pool_size=(8)))

    model.add(Conv1D(128, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 8, padding='same'))
    model.add(Activation('relu'))

    model.add(Conv1D(128, 8, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(MaxPooling1D(pool_size=(8)))

    model.add(Conv1D(64, 8, padding='same'))
    model.add(Activation('relu'))

    model.add(Conv1D(64, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Flatten())

    model.add(Dense(8)) # Target class number, in this case the 8 emotions
    model.add(Activation('softmax'))
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model