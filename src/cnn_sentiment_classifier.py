'''
Created on 23 feb. 2018

Ejemplo de clasificador opniones a nivel de oración (documento) usando una CNN,
y utilizando la librería TensorFlow. 
 
@author: Eugenio Martínez Cámara
@organization: Universida de Granada
@requires: TensorFlow, Pythnon3, NLTK, sentence_polarity corpus de NLTK
'''

#Variables globales
RANDOM_SEED = 7

import random
import nltk.corpus.sentence_polarity as sent_pol
from sklearn.model_selection import train_test_split

def data_preparation_train_test():
    """Generación del conjunto de entrenamiento y test
    """
    #Esto lo hago porque a travé de la documentación de NLTK sé como es el corpus.
    n_pos_sents = len(sent_pol.sents(categories="pol"))
    n_neg_sents = len(sent_pol.sents(categories="neg"))
    #db_indexes: Cada posición se corresponde con una oración del corpus
    db_indexes = [i for i in range(n_pos_sents + n_neg_sents)]
    #db_labels: Cada posición se corresponde con una etiqueta de opinión del corpus.
    #Cada posición de esta lista se corresponde con cada posición de db_indexes.
    db_labels = [1] * n_pos_sents + [0] * n_neg_sents
    train_test_split(db_indexes, db_labels,test_size=0.2,shuffle=True, stratify=[1,0])
    

if __name__ == '__main__':
    
    #Definir semilla aleatoria
    random.seed(RANDOM_SEED)
    
    #2.- Leer corpus y partición de entrenamiento y test.