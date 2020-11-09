from fitness.base_ff_classes.base_ff import base_ff
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
import tensorflow as tf
import re
import os


class paper(base_ff):

    maximise = True

    def __init__(self):
        super().__init__()
        
    def evaluate(self, ind, **kwargs):
        
        print('\nFENOTIPO: %s\n' % ind.phenotype)

        # Capturando os parametros do fenotipo
        nconv, npool, nfc = [int(i) for i in re.findall('\d+', ind.phenotype)]
        has_pool = 'pool' in ind.phenotype

        # Carregando dataset
        (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

        validation_images, test_images, validation_labels, test_labels = train_test_split(test_images, test_labels, test_size=0.33, random_state=42)

        # Normalizando
        train_images = train_images / 255.0
        test_images = test_images / 255.0
        validation_images = validation_images / 255.0

        model_name = 'conv_%d-pool_%d-fc_%d-haspool_%s' % (nconv, npool, nfc, has_pool)
        path = '/pesquisa/trained_models/%s' % model_name

        # num de filtros
        filter_size = 32
        nfilter = 1 # comeca com 1 pois logo abaixo ja adiciono uma convolucao

        # Iniciando o modelo da RN
        model = models.Sequential()
        model.add(layers.Conv2D(filter_size, (3, 3), activation='relu', input_shape=(32, 32, 3)))

        try:
            
            # Pooling
            for i in range(npool):
                n = nconv if i != 0 else nconv - 1
        
                # Convolucoes
                for j in range(n):
        
                    # Numero de filtros duplica a cada duas Convolucoes
                    if nfilter == 2:
                        filter_size *= 2
                        nfilter = 0
        
                    model.add(layers.Conv2D(filter_size, (3, 3), activation='relu', padding='same'))
                    nfilter += 1

                # Adiciona o pooling somente se estiver no fenotipo.
                if has_pool:
                    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

            model.add(layers.Flatten())

            # fully connected
            for i in range(nfc):
                model.add(layers.Dense(256, activation='relu'))

            # (((conv*3)None)*1)fc*0
            # (((conv*2)pool)*3)fc*2
            model.add(layers.Dense(10, activation='softmax'))
            model.summary()

        except Exception as ex:
            print(ex)
            return 0

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        if os.path.isfile('%s.index' % path):
            print('Model já foi treinado. Carrengando pesos...')
            model.load_weights(path)
        else:
            print('Model ainda não foi treinado. Treinando...')
            model.fit(train_images, train_labels, epochs=70, batch_size=128, 
                validation_data=(validation_images, validation_labels))
            model.save_weights(path)

        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

        return test_acc