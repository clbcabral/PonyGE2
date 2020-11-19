from tensorflow.keras import datasets, layers, models, callbacks, optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import re


def evaluate(phenotype):
        
    print('FENOTIPO: %s' % phenotype)

    nconv, npool, nfc, nfcneuron = [int(i) for i in re.findall('\d+', phenotype.split('lr-')[0])]
    has_dropout = 'dropout' in phenotype
    has_batch_normalization = 'bnorm' in phenotype
    has_pool = 'pool' in phenotype
    learning_rate = float(phenotype.split('lr-')[1])

    # Carregando dataset
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    validation_images, test_images, validation_labels, test_labels = train_test_split(test_images, test_labels, test_size=0.33, random_state=42)

    # Normalizando
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    validation_images = validation_images / 255.0

    lb = LabelBinarizer()
    train_labels = lb.fit_transform(train_labels)
    validation_labels = lb.transform(validation_labels)
    test_labels = lb.transform(test_labels)

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
        model.summary()

    except Exception as ex:
        print(ex)
        return 0

    opt = optimizers.Adam(lr=learning_rate)
    
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        
    es = callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=35, baseline=0.5)
    
    model.fit(train_images, train_labels, epochs=70, batch_size=128, 
        validation_data=(validation_images, validation_labels), callbacks=[es])
        
    _, acuracia = model.evaluate(test_images, test_labels, verbose=2)
    
    return acuracia


evaluate('(((conv*2)bnorm-pool-)*3)fc*2*256*lr-0.01')