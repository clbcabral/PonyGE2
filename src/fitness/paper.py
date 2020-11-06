from fitness.base_ff_classes.base_ff import base_ff
from tensorflow.keras import datasets, layers, models
import tensorflow as tf
import re


class paper(base_ff):

    def __init__(self):
        super().__init__()
        
    def evaluate(self, ind, **kwargs):
        
        print('\nFENOTIPO: %s\n' % ind.phenotype)

        # Capturando os parametros do fenotipo
        nconv, npool, nfc = [int(i) for i in re.findall('\d+', ind.phenotype)]
        has_pool = 'pool' in ind.phenotype
        
        # num de filtros
        filter_size = 32
        nfilter = 1 # comeca com 1 pois logo abaixo ja adiciono uma convolucao

        # Carregando dataset
        (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

        # Normalizando
        train_images = train_images / 255.0
        test_images = test_images / 255.0

        # Dividindo dataset em validacao (80%) e testes (20%)
        validation_images = test_images[:8000]
        validation_labels = test_labels[:8000]
        test_images = test_images[8000:]
        test_labels = test_labels[8000:]

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
        
                    model.add(layers.Conv2D(filter_size, (3, 3), activation='relu'))
                    nfilter += 1

                # Adiciona o pooling somente se estiver no fenotipo.
                if has_pool:
                    model.add(layers.MaxPooling2D((2, 2)))

            # fully connected
            for i in range(nfc):
                model.add(layers.Dense(256, activation='relu'))

            # (((conv*3)None)*1)fc*0
            # (((conv*2)pool)*3)fc*2
            model.add(layers.Flatten())
            model.add(layers.Dense(10, activation='softmax'))
            model.summary()

        except Exception as ex:
            print(ex)
            return 0

        adam = tf.keras.optimizers.Adam(learning_rate=0.01)

        model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

        model.fit(train_images, train_labels, epochs=70, validation_data=(validation_images, validation_labels))

        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

        model_name = 'conv_%d-pool_%d-fc_%d-haspool_%s' % (nconv, npool, nfc, has_pool)

        model.save('/pesquisa/trained_models/%s' % model_name)

        return test_acc