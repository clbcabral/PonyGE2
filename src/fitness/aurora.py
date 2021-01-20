from fitness.base_ff_classes.base_ff import base_ff
from tensorflow.keras import Input
from tensorflow.keras import datasets, layers, models, callbacks, optimizers
from tensorflow.keras import backend as K 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import re, csv, os, random


class aurora(base_ff):

    maximise = True

    def __init__(self):
        super().__init__()
        self.filename = '/pesquisa/aurora.csv'

    def get_metrics(self, phenotype):
        accuracy, accuracy_sd = None, None
        with open(self.filename, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[0] == phenotype:
                    accuracy = float(row[1])
                    accuracy_sd = float(row[2])
                    break
        return accuracy, accuracy_sd

    def save_metrics(self, phenotype, accuracy, accuracy_sd):
        with open(self.filename, mode='a') as file:
            writer = csv.writer(file)
            writer.writerow([phenotype, accuracy, accuracy_sd])

    def load_data(self):
        # Load dataset
        (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
        validation_images, test_images, validation_labels, test_labels = train_test_split(test_images, test_labels, test_size=0.2, random_state=42)
        
        # Normalizing
        train_images = train_images.astype("float") / 255.0
        test_images = test_images.astype("float") / 255.0
        validation_images = validation_images.astype("float") / 255.0

        lb = LabelBinarizer()
        train_labels = lb.fit_transform(train_labels)
        validation_labels = lb.transform(validation_labels)
        test_labels = lb.transform(test_labels)
        
        return train_images, train_labels, test_images, test_labels, validation_images, validation_labels

    def build_model(self, phenotype):

        # To free memory on google colab.
        if K.backend() == 'tensorflow':
            K.clear_session()

        model = models.Sequential()
        model.add(Input(shape=(32, 32, 3)))

        layers_list = phenotype.split('+')

        for layer in layers_list:
            params = layer.strip().split(' ')
            if layer.startswith('Conv2D'):
                _, filters, ka, kb = [int(i) for i in re.findall('\d+', layer)]
                model.add(layers.Conv2D(filters, (ka, kb), activation=params[-1]))
            elif layer.startswith('Dropout'):
                model.add(layers.Dropout(float(params[-1])))
            elif layer.startswith('MaxPooling2D'):
                _, pa, pb = [int(i) for i in re.findall('\d+', layer)]
                model.add(layers.MaxPooling2D(pool_size=(pa, pb), padding=params[-1]))
            elif layer.startswith('AvgPooling2D'):
                _, pa, pb = [int(i) for i in re.findall('\d+', layer)]
                model.add(layers.AveragePooling2D(pool_size=(pa, pb), padding=params[-1]))

        model.add(layers.Flatten())

        for layer in layers_list:
            params = layer.strip().split(' ')
            if layer.startswith('Dense'):
                model.add(layers.Dense(int(params[-1])))

        model.add(layers.Dense(10, activation='softmax'))

        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def train_model(self, model):

        accuracies = []

        train_images, train_labels, test_images, \
            test_labels, validation_images, validation_labels = self.load_data()

        # Train three times
        for i in range(3):

            print('Trainning %s of 3' % (i + 1))

            # Early Stop when bad networks are identified        
            es = callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10, baseline=0.5)

            model.fit(train_images, train_labels, epochs=50, batch_size=128, 
                validation_data=(validation_images, validation_labels), callbacks=[es])
            
            loss, accuracy = model.evaluate(test_images, test_labels, verbose=1)

            accuracies.append(accuracy)

            if i == 0 and accuracy < 0.5:
                break

        return np.mean(accuracies), np.std(accuracies)

    def evaluate(self, ind, **kwargs):

        print('PHENOTYPE: %s' % ind.phenotype)

        accuracy, accuracy_sd = self.get_metrics(ind.phenotype)

        if accuracy is None:

            print('Phenotype not yet trained. Building...')

            try:
                model = self.build_model(ind.phenotype)
                accuracy, accuracy_sd = self.train_model(model)
            except Exception as e:
                print(e)
                accuracy, accuracy_sd = 0.0, 0.0, 0.0, 0.0

            self.save_metrics(ind.phenotype, accuracy, accuracy_sd)

        print(accuracy, accuracy_sd)

        return accuracy