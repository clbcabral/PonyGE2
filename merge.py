import sys, csv

arquivos = sys.argv[1:]

mapa = dict()

for arquivo in arquivos:
    print(arquivo)
    with open(arquivo, mode='r') as aqv:
        leitor = csv.DictReader(aqv)
        for linha in leitor:
            fenotipo = linha['phenotype']
            acuracia = linha['accuracy']
            f1_score = linha['f1_score']
            if fenotipo not in mapa or acuracia > mapa[fenotipo][0]:
                mapa[fenotipo] = (acuracia, f1_score)

with open('resultado.csv', mode='w+') as aqv:
    escritor = csv.DictWriter(aqv, fieldnames=['phenotype', 'accuracy', 'f1_score'])
    escritor.writeheader()
    for fenotipo in mapa:
        acuracia, f1_score = mapa[fenotipo]
        escritor.writerow({'phenotype': fenotipo, 'accuracy': acuracia, 'f1_score': f1_score})