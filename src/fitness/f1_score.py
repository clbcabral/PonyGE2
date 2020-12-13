from fitness.base_ff_classes.base_ff import base_ff
import csv


class f1_score(base_ff):

    maximise = True
    
    def __init__(self):
        super().__init__()
        self.filename = '/pesquisa/phenotypes.csv'

    def get_metrics(self, phenotype):
        accuracy, f1_score = None, None
        with open(self.filename, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[0] == phenotype:
                    accuracy = float(row[1])
                    f1_score = float(row[2])
                    break
        return accuracy, f1_score
    
    def evaluate(self, ind, **kwargs):
        print('PHENOTYPE: %s' % ind.phenotype)
        _, f1_score = self.get_metrics(ind.phenotype)
        return f1_score
