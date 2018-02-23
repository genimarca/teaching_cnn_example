'''
Created on 23 feb. 2018

Ejemplo de clasificador opniones a nivel de oración (documento) usando una CNN,
y utilizando la librería TensorFlow.

Se utilizará el corpus sentence_polarity, que en concreto es el corpus de Bing Liu.
NLTK lo ofrece segmentado en oraciones y tokenizado. 
 
@author: Eugenio Martínez Cámara
@organization: Universida de Granada
@requires: TensorFlow, Pythnon3, NLTK, sentence_polarity corpus de NLTK
'''

import random
import nltk.corpus.sentence_polarity as sent_pol
from sklearn.model_selection import train_test_split
import tensorflow as tf
#Variables globales
RANDOM_SEED = 7
MAX_LENGTH = 65

def data_preparation_train_test():
    """Generación del conjunto de entrenamiento y test
    
    Returns:
        train_sents: es una lista con las oraciones (listas de palabras) para el entrenamiento.
        test_sents: es una lista con las oraciones (listas de palabras) para el test.
        train_labels: lista de enteros con la clase (0:negativo, 1:positivo) de 
        las oraciones de entrenamiento.
        test_labels: lista de enteros con la clase (0:negativo, 1:positivo) de 
        las oraciones de test.
    """
    #Esto lo hago porque a travé de la documentación de NLTK sé como es el corpus.
    n_pos_sents = len(sent_pol.sents(categories="pol"))
    n_neg_sents = len(sent_pol.sents(categories="neg"))
    #db_indexes: Cada posición se corresponde con una oración del corpus
    db_indexes = [i for i in range(n_pos_sents + n_neg_sents)]
    #db_labels: Cada posición se corresponde con una etiqueta de opinión del corpus.
    #Cada posición de esta lista se corresponde con cada posición de db_indexes.
    db_labels = [1] * n_pos_sents + [0] * n_neg_sents
    train_indexes, test_indexes, train_labels, test_labels = train_test_split(db_indexes, db_labels,test_size=0.2,shuffle=True, stratify=db_labels)
    
    train_sents = [db_indexes[i] for i in train_indexes]
    test_sents = [db_indexes[i] for i in test_indexes]
    
    return (train_sents, test_sents, train_labels, test_labels)
    
def nn_graph():
    
    tf.placeholder(tf.float32, shape=[None, MAX_LENGTH], name="input_sentence")



if __name__ == '__main__':
    
    #Definir semilla aleatoria
    random.seed(RANDOM_SEED)
    
    #2.- Leer corpus y partición de entrenamiento y test.
    train_sents, test_sents, train_labels, test_labels = data_preparation_train_test()
    
    