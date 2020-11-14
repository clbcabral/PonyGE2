from fitness.base_ff_classes.base_ff import base_ff
from tensorflow.keras import datasets, layers, models, callbacks
from sklearn.model_selection import train_test_split
import re
import os


class paper(base_ff):

    maximise = True

    def __init__(self):
        super().__init__()
        
    def evaluate(self, ind, **kwargs):
        
        print('FENOTIPO: %s' % ind.phenotype)
        
        # Capturando os parametros do fenotipo
        nconv, npool, nfc = [int(i) for i in re.findall('\d+', ind.phenotype)]
        has_dropout = 'dropout' in ind.phenotype
        has_batch_normalization = 'bnorm' in ind.phenotype
        has_pool = 'pool' in ind.phenotype
        is_maxpool = 'max' in ind.phenotype

        # Carregando dataset
        (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

        validation_images, test_images, validation_labels, test_labels = train_test_split(test_images, test_labels, test_size=0.33, random_state=42)

        # Normalizando
        train_images = train_images / 255.0
        test_images = test_images / 255.0
        validation_images = validation_images / 255.0

        model_name = 'conv_%d-pool_%d-fc_%d-haspool_%s-dropout_%s-bnorm_%s-ismax_%s' % (nconv, npool, nfc, has_pool, has_dropout, has_batch_normalization, is_maxpool)
        path = '/pesquisa/trained_models/%s' % model_name

        # num de filtros
        filter_size = 32
        nfilter = 1 # comeca com 1 pois logo abaixo ja adiciono uma convolucao

        # Iniciando o modelo da RN
        model = models.Sequential()
        model.add(layers.Conv2D(filter_size, (3, 3), activation='relu', input_shape=(32, 32, 3)))

        try:

            if has_batch_normalization:
                model.add(layers.BatchNormalization())
            
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

                    if has_batch_normalization:
                        model.add(layers.BatchNormalization())

                    nfilter += 1

                # Adiciona o pooling somente se estiver no fenotipo.
                if has_pool:
                    if is_maxpool:
                        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
                    else:
                        model.add(layers.AvgPool2D(pool_size=(2, 2)))
                    if has_dropout:
                        model.add(layers.Dropout(0.25))

            model.add(layers.Flatten())

            # fully connected
            for i in range(nfc):
                model.add(layers.Dense(256, activation='relu'))

            # FENOTIPO: (((conv*1)bnorm-max-pool-)*1)fc*0
            # FENOTIPO: (((conv*1)bnorm-max-pool-)*2)fc*0
            # FENOTIPO: (((conv*1)bnorm-)*2)fc*0
            # FENOTIPO: (((conv*3))*3)fc*2
            model.add(layers.Dense(10, activation='softmax'))
            model.summary()

        except Exception as ex:
            print(ex)
            return 0

        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        if os.path.isfile('%s.index' % path):
            print('Model já foi treinado. Carrengando pesos...')
            model.load_weights(path)
        else:
            print('Model ainda não foi treinado. Treinando...')
            es = callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=35)
            model.fit(train_images, train_labels, epochs=70, batch_size=128, 
                validation_data=(validation_images, validation_labels), callbacks=[es])
            model.save_weights(path)

        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

        return test_acc