# version 2.1 du 21 mai 2022

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from seaborn import heatmap
from sklearn.metrics import confusion_matrix
    

def plot_loss_accuracy(hist):
    '''Plot training & validation loss & accuracy values, giving an argument
       'hist' of type 'tensorflow.python.keras.callbacks.History'. '''
    
    custom_lines = [Line2D([0], [0], color='blue', lw=1, marker='o'),
                    Line2D([0], [0], color='orange', lw=1, marker='o')]
    
    plt.figure(figsize=(15,5))
    
    if not isinstance(hist, list): hist = [hist]
        
    ax1 = plt.subplot(1,2,1)
    for h in hist:
        if h.history.get('accuracy'):
            ax1.plot(np.array(h.epoch)+1, h.history['accuracy'], 'o-', color='blue')
        if h.history.get('val_accuracy'):
            ax1.plot(np.array(h.epoch)+1, h.history['val_accuracy'], 'o-', color='orange')
    ax1.set_title('Model accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch') 
    ax1.set_xticks(np.arange(1, len(h.epoch)+1))
    ax1.grid()
    ax1.legend(custom_lines, ['Train', 'Test'])
    
    # Plot training & validation loss values
    ax2 = plt.subplot(1,2,2)
    for h in hist:
        if h.history.get('loss'):
            ax2.plot(np.array(h.epoch)+1, h.history['loss'], 'o-', color='blue')
        if h.history.get('val_loss'):
            ax2.plot(np.array(h.epoch)+1, h.history['val_loss'], 'o-', color='orange')
    ax2.set_title('Model loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_xticks(np.arange(1, len(h.epoch)+1))
    ax2.grid()
    ax2.legend(custom_lines, ['Train', 'Test'])

    plt.show()

def plot_images(image_array, r, L, C):
    '''Plot the images of image_array on a grid L x C, starting at
       rank r'''
    plt.figure(figsize=(C,L))
    for i in range(L*C):
        plt.subplot(L, C, i+1)
        plt.imshow(image_array[r+i], cmap='gray')
        plt.xticks([]); plt.yticks([])


def show_cm(true, results, classes):
    ''' true  : the actual labels 
        results : the labels computed by the trained network (one-hot format)
        classes : list of possible label values'''
    predicted = np.argmax(results, axis=-1) # tableau d'entiers entre 0 et 9 
    cm = confusion_matrix(true, predicted)
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize=(11,9))
    heatmap(df_cm, annot=True, cbar=False, fmt="3d")
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.show()
    
