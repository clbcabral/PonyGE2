from algorithm.parameters import params
from fitness.base_ff_classes.base_ff import base_ff
from tensorflow.keras import datasets, layers, models, callbacks, optimizers
from tensorflow.keras import backend as K 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import re, csv, os, requests
import tensorflow as tf


class paper_eurosat_tpu(base_ff):

    maximise = True
    multi_objective = True

    def __init__(self):
        super().__init__()
        self.num_obj = 2
        fit = base_ff()
        fit.maximise = True
        self.fitness_functions = [fit, fit]
        self.default_fitness = [float('nan'), float('nan')]
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
        tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
        self.tpu_strategy = tpu_strategy

    def load_data(self):
        
        dsnp = np.load('/content/%s.npz' % params['DATASET_NAME'])

        images = dsnp['images']
        labels = dsnp['labels']

        images = images.reshape((images.shape[0], 64, 64, 3))
        images = images.astype("float") / 255.0

        train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
        validation_images, test_images, validation_labels, test_labels = train_test_split(test_images, test_labels, test_size=0.2, random_state=42)

        lb = LabelBinarizer()

        train_labels = lb.fit_transform(train_labels)
        validation_labels = lb.transform(validation_labels)
        test_labels = lb.transform(test_labels)

        return train_images, train_labels, test_images, test_labels, validation_images, validation_labels

    def get_metrics(self, phenotype):

        accuracy, accuracy_sd, f1_score, f1_score_sd = None, None, None, None
  
        r = requests.get(params['METRICS_URL'], params={
            'dataset': params['DATASET_NAME'],
            'phenotype': phenotype,
        })
        data = r.json()

        if len(data):
            data = data[0]
            accuracy = float(data['accuracy'])
            accuracy_sd = float(data['accuracy_sd'])
            f1_score = float(data['f1_score'])
            f1_score_sd = float(data['f1_score_sd'])
    
        return accuracy, accuracy_sd, f1_score, f1_score_sd

    def save_metrics(self, phenotype, accuracy, accuracy_sd, f1_score, f1_score_sd):
        data = {
            'dataset': params['DATASET_NAME'],
            'phenotype': phenotype,
            'accuracy': accuracy,
            'accuracy_sd': accuracy_sd,
            'f1_score': f1_score,
            'f1_score_sd': f1_score_sd,
        }
        r = requests.post(params['METRICS_URL'], json=data)

    def build_model(self, phenotype):

        with self.tpu_strategy.scope():

            nconv, npool, nfc, nfcneuron = [int(i) for i in re.findall('\d+', phenotype.split('lr-')[0])]
            has_dropout = 'dropout' in phenotype
            has_batch_normalization = 'bnorm' in phenotype
            has_pool = 'pool' in phenotype
            learning_rate = float(phenotype.split('lr-')[1])

            # number of filters
            filter_size = 32

            model = models.Sequential()
            model.add(layers.InputLayer(input_shape=(64, 64, 3)))

            # Pooling
            for i in range(npool):

                # Convolutions
                for j in range(nconv):

                    model.add(layers.Conv2D(filter_size, (3, 3), activation='relu', padding='same'))

                    # Duplicate number of filters for each two convolutions
                    if (((i + j) % 2) == 1): filter_size = filter_size * 2

                    # Add batch normalization
                    if has_batch_normalization:
                        model.add(layers.BatchNormalization())

                # Add pooling
                if has_pool:
                    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
                    # Add dropout
                    if has_dropout:
                        model.add(layers.Dropout(0.25))

            model.add(layers.Flatten())

            # fully connected
            for i in range(nfc):
                model.add(layers.Dense(nfcneuron))
                model.add(layers.Activation('relu'))

            if has_dropout:
                model.add(layers.Dropout(0.5))

            model.add(layers.Dense(10, activation='softmax'))
            model.summary()

            opt = optimizers.Adam(learning_rate=learning_rate)

            # F1 Score metric function
            def f1_score(y_true, y_pred):
                true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
                possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
                predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
                precision = true_positives / (predicted_positives + K.epsilon())
                recall = true_positives / (possible_positives + K.epsilon())
                f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
                return f1_val

            model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', f1_score])

        return model


    def train_model(self, model):

        batch_size = 128

        accuracies, f1_scores = [], []

        train_images, train_labels, test_images, \
            test_labels, validation_images, validation_labels = self.load_data()

        train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(batch_size, drop_remainder=True)
        validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels)).batch(batch_size, drop_remainder=True)

        # Train three times
        for i in range(3):

            # To free memory on google colab.
            if K.backend() == 'tensorflow':
                K.clear_session()

            print('Trainning %s of 3' % (i + 1))

            # Early Stop when bad networks are identified        
            es = callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10, baseline=0.5)

            model.fit(train_ds,
                epochs=70, 
                batch_size=batch_size, 
                verbose=1,
                validation_data=validation_ds,
                callbacks=[es])
            
            loss, accuracy, f1_score = model.evaluate(test_images, test_labels, verbose=1)

            accuracies.append(accuracy)
            f1_scores.append(f1_score)

            if i == 0 and accuracy < 0.5:
                break

        return np.mean(accuracies), np.std(accuracies), np.mean(f1_scores), np.std(f1_scores)

    def evaluate(self, ind, **kwargs):

        print('PHENOTYPE: %s' % ind.phenotype)

        accuracy, accuracy_sd, f1_score, f1_score_sd = self.get_metrics(ind.phenotype)

        if accuracy is None and f1_score is None:

            print('Phenotype not yet trained. Building...')

            with self.tpu_strategy.scope():
                model = self.build_model(ind.phenotype)

            accuracy, accuracy_sd, f1_score, f1_score_sd = self.train_model(model)

            self.save_metrics(ind.phenotype, accuracy, accuracy_sd, f1_score, f1_score_sd)

        print(accuracy, accuracy_sd, f1_score, f1_score_sd)

        return accuracy, f1_score

    @staticmethod
    def value(fitness_vector, objective_index):
        """
        This is a static method required by NSGA-II for sorting populations
        based on a given fitness function, or for returning a given index of a
        population based on a given fitness function.

        :param fitness_vector: A vector/list of fitnesses.
        :param objective_index: The index of the desired fitness.
        :return: The fitness at the objective index of the fitness vecror.
        """

        if not isinstance(fitness_vector, list):
            return float("inf")

        return fitness_vector[objective_index]
