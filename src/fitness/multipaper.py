from fitness.base_ff_classes.base_ff import base_ff
from fitness.paper import paper
from tensorflow.keras import datasets, layers, models, callbacks, optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K 
from sklearn.model_selection import train_test_split
import numpy as np
import re


class multipaper(base_ff):
    """
    An example of a single fitness class that generates
    two fitness values for multiobjective optimisation
    """

    maximise = True
    multi_objective = True

    def __init__(self):

        # Initialise base fitness function class.
        super().__init__()

        self.arquivo = '/pesquisa/fenotipos.csv'

        # Set list of individual fitness functions.
        self.num_obj = 2
        fit = paper()
        fit.maximise = True
        self.fitness_functions = [fit, fit]
        self.default_fitness = [float('nan'), float('nan')]


    def evaluate(self, ind, **kwargs):
        
        print(ind.phenotype)
    
        if K.backend() == 'tensorflow':
            K.clear_session()

        nconv, npool, nfc, nfcneuron = [int(i) for i in re.findall('\d+', ind.phenotype.split('lr-')[0])]
        has_dropout = 'dropout' in ind.phenotype
        has_batch_normalization = 'bnorm' in ind.phenotype
        has_pool = 'pool' in ind.phenotype
        learning_rate = float(ind.phenotype.split('lr-')[1])

        # Carregando dataset
        (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

        validation_images, test_images, validation_labels, test_labels = train_test_split(test_images, test_labels, test_size=0.33, random_state=42)

        # Normalizando
        train_images = train_images / 255.0
        test_images = test_images / 255.0
        validation_images = validation_images / 255.0

        train_labels = to_categorical(train_labels)
        validation_labels = to_categorical(validation_labels)
        test_labels = to_categorical(test_labels)

        # num de filtros
        filter_size = 32
        nfilter = 0

        # Iniciando o modelo da RN
        model = models.Sequential()

        try:
        
            # Pooling
            for i in range(npool):
        
                # Convolucoes
                for j in range(nconv):
        
                    model.add(layers.Conv2D(filter_size, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))

                    nfilter += 1

                    # Numero de filtros duplica a cada duas Convolucoes
                    if nfilter == 2:
                        filter_size *= 2
                        nfilter = 0

                    if has_batch_normalization:
                        model.add(layers.BatchNormalization())

                # Adiciona o pooling somente se estiver no fenotipo.
                if has_pool:
                    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
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
            #model.summary()

        except Exception as ex:
            print(ex)
            return 0

        opt = optimizers.Adam(lr=learning_rate)

        def f1_score(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            recall = true_positives / (possible_positives + K.epsilon())
            f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
            return f1_val
        
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', f1_score])
        
        es = callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10, baseline=0.5)

        model.fit(train_images, train_labels, epochs=70, batch_size=128, 
            validation_data=(validation_images, validation_labels), callbacks=[es])
            
        loss, acuracia, f1 = model.evaluate(test_images, test_labels, verbose=0)

        print(acuracia, f1)

        return (acuracia, f1)

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
