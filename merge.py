import sys, csv

arquivos = sys.argv[1:]

mapa = dict()

for arquivo in arquivos:
    print(arquivo)
    with open(arquivo, mode='r') as aqv:
        leitor = csv.DictReader(aqv)
        for linha in leitor:
            fenotipo = linha['fenotipo']
            acuracia = linha['acuracia']
            if fenotipo not in mapa or acuracia > mapa[fenotipo]:
                mapa[fenotipo] = acuracia

with open('resultado.csv', mode='w+') as aqv:
    escritor = csv.DictWriter(aqv, fieldnames=['fenotipo', 'acuracia'])
    escritor.writeheader()
    for fenotipo in mapa:
        acuracia = mapa[fenotipo]
        escritor.writerow({'fenotipo': fenotipo, 'acuracia': acuracia})