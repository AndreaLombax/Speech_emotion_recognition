from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
from model import get_model
from dataset import RAVDESS_Dataset
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.config import list_physical_devices


if __name__ == "__main__":
    print("Num GPUs Available: ", len(list_physical_devices('GPU')))

    RAVDESS_Dataset_class = RAVDESS_Dataset()
    X, Y = RAVDESS_Dataset_class.create_dataset()

    # As this is a multiclass classification problem onehotencoding our Y.
    encoder = OneHotEncoder()
    Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()

    # splitting data
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, train_size=0.70, random_state=0, shuffle=True)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, train_size=0.5, random_state=0, shuffle=True)
    print("X train:" ,x_train.shape)
    print("Y train:" ,y_train.shape)
    print("X test:" ,x_test.shape)
    print("Y test:" ,y_test.shape)
    print("X val:" ,x_val.shape)
    print("X val:" ,y_val.shape)

    # scaling our data with sklearn's Standard scaler
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    x_val = scaler.transform(x_val)
    x_train.shape, y_train.shape, x_test.shape, y_test.shape

    # making our data compatible to model.
    x_train = np.expand_dims(x_train, axis=2)
    x_test = np.expand_dims(x_test, axis=2)
    x_val = np.expand_dims(x_val, axis=2)
    x_train.shape, y_train.shape, x_test.shape, y_test.shape

    model = get_model(X.shape[1])

    epochs = 600
    rlrp = ReduceLROnPlateau(monitor='loss', factor=0.8, verbose=1, patience=15, min_lr=0.000001) #
    history=model.fit(x_train, y_train, batch_size=64, epochs=epochs, validation_data=(x_val, y_val), callbacks=[rlrp])

    model.save('model.h5')

    #-------------------- TEST --------------------#
    model = load_model('model.h5')

    print("Accuracy of our model on test data : " , model.evaluate(x_test,y_test)[1]*100 , "%")

    epochs = [i for i in range(epochs)]
    fig , ax = plt.subplots(1,2)
    train_acc = history.history['accuracy']
    train_loss = history.history['loss']
    test_acc = history.history['val_accuracy']
    test_loss = history.history['val_loss']

    fig.set_size_inches(20,8)
    ax[0].plot(epochs , train_loss , label = 'Training Loss')
    ax[0].plot(epochs , test_loss , label = 'Testing Loss')
    ax[0].set_title('Training & Testing Loss')
    ax[0].legend()
    ax[0].set_xlabel("Epochs")

    ax[1].plot(epochs , train_acc , label = 'Training Accuracy')
    ax[1].plot(epochs , test_acc , label = 'Testing Accuracy')
    ax[1].set_title('Training & Testing Accuracy')
    ax[1].legend()
    ax[1].set_xlabel("Epochs")
    plt.savefig('train_loss_accuracy.png')

    
    pred_test = model.predict(x_test)
    y_pred = encoder.inverse_transform(pred_test)

    y_test = encoder.inverse_transform(y_test)

    df = pd.DataFrame(columns=['Predicted Labels', 'Actual Labels'])
    df['Predicted Labels'] = y_pred.flatten()
    df['Actual Labels'] = y_test.flatten()

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize = (12, 10))
    cm = pd.DataFrame(cm , index = [i for i in encoder.categories_] , columns = [i for i in encoder.categories_])
    sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
    plt.title('Confusion Matrix', size=20)
    plt.xlabel('Predicted Labels', size=14)
    plt.ylabel('Actual Labels', size=14)
    plt.savefig('confusion_matrix.png')