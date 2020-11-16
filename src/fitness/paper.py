from fitness.base_ff_classes.base_ff import base_ff
from tensorflow.keras import datasets, layers, models, callbacks, optimizers
from sklearn.model_selection import train_test_split
from datetime import datetime
import re, csv, os


class paper(base_ff):

    maximise = True

    def __init__(self):
        self.arquivo = datetime.now().strftime('../results/fenotipos-%d%m%Y-%H%M%S.csv')
        with open(self.arquivo, mode='w+') as file:
            writer = csv.DictWriter(file, fieldnames=['fenotipo', 'acuracia'])
            writer.writeheader()
        super().__init__()

    def evaluate(self, ind, **kwargs):
        
        print('FENOTIPO: %s' % ind.phenotype)

        nconv, npool, nfc = [int(i) for i in re.findall('\d+', ind.phenotype)]
        has_dropout = 'dropout' in ind.phenotype
        has_batch_normalization = 'bnorm' in ind.phenotype
        has_pool = 'pool' in ind.phenotype

        # Carregando dataset
        (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

        validation_images, test_images, validation_labels, test_labels = train_test_split(test_images, test_labels, test_size=0.33, random_state=42)

        # Normalizando
        train_images = train_images / 255.0
        test_images = test_images / 255.0
        validation_images = validation_images / 255.0

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
                    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
                    if has_dropout:
                        model.add(layers.Dropout(0.25))

            model.add(layers.Flatten())

            # fully connected
            for i in range(nfc):
                model.add(layers.Dense(256, activation='relu'))

            model.add(layers.Dense(10, activation='softmax'))
            model.summary()

        except Exception as ex:
            print(ex)
            return 0

        opt = optimizers.Adam()
        
        model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        acuracia = None

        with open(self.arquivo, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[0] == ind.phenotype:
                    acuracia = row[1]
                    break
        
        if not acuracia:

            print('Model ainda não foi treinado. Treinando...')
            
            es = callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=35)
            model.fit(train_images, train_labels, epochs=70, batch_size=128, 
                validation_data=(validation_images, validation_labels), callbacks=[es])
                
            _, acuracia = model.evaluate(test_images, test_labels, verbose=2)

            with open(self.arquivo, mode='a') as file:
                writer = csv.writer(file)
                writer.writerow([ind.phenotype, 0.0])

        return acuracia