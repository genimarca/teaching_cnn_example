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
EMBEDDINGS_DIMENSIONS = 300
KERNEL_SIZE = 2
CNN_OUTPUT_FEATURES = 100
EPOCHS = 10
SIZE_BATCHES = 50
OOV_INDEX = 0


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
    n_pos_sents = len(sent_pol.sents(categories="pos"))
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
    
def build_vocabulary(input_corpus, index_start):
    """Genera un vocabulario a partir de un conjunto de oraciones/documentos de
    entrada.
    
    En este caso, las oraciones/documentos deben estar tokenizdos.
    
    Args:
        input_corpus: Lista de listas de oraciones tokenizadas.
        index_start: interger with the first value of the index of the vocabulary.
    """
    vocabulary = {}
    own_lower = str.lower
    index = index_start
    for sent in input_corpus:
        for word in sent:
            word = own_lower(word)
            if word not in vocabulary:
                vocabulary[word] = index
                index += 1
    return vocabulary
                
            
    

def nn_graph():
    """Definición del grafo de la red neuronal.
    """
    
    #Entrada
    x_sentences = tf.placeholder(tf.float32, shape=[None, MAX_LENGTH], name="input_sentence")
    y_labels = tf.placeholder(tf.float32, shape=[None,1], name="input_label")
    
    #Capa de embeddings. Aquí generamos los embeddgins de manera aleatoria. En el trabajo 
    #se utilizarán unos embeddings pre-entrenados.
    word_embeddings = tf.get_variable("word_embeddings", shape=[MAX_LENGTH, EMBEDDINGS_DIMENSIONS], dtype=tf.float32, trainable=True)
    x_sentences_embeddings = tf.nn.embedding_lookup(word_embeddings, x_sentences, name="layer_embeddings_lookup")
    
    #CNN
    x_sentences_conv_activation = None
    with tf.variable_scope("cnn_layer") as scope:
        v_kernel = tf.get_variable("kernel", shape=[2,EMBEDDINGS_DIMENSIONS, CNN_OUTPUT_FEATURES], dtype=tf.float32)
        x_sentences_conv = tf.nn.conv1d(x_sentences_embeddings, v_kernel, 1, padding="VALID", name="cnn_operation")
        v_bias = tf.get_variable("bias",shape=[CNN_OUTPUT_FEATURES], dtype=tf.float32, initializer=tf.constant(0.1))
        pre_activation = tf.nn.bias_add(x_sentences_conv, v_bias)
        x_sentences_conv_activation = tf.nn.tanh(pre_activation, name="cnn_activation")
    
    #Full connect layer
    x_sentences_dense_activation = None
    with tf.variable_scope("dense_layer") as scope:
        weights = tf.get_variable("dense_weigths", shape=[None, x_sentences_conv_activation.get_shape()[2].value, x_sentences_conv_activation.get_shape()[2].value], dtype=tf.float32)
        bias = tf.get_variable("dense_variables", shape=[x_sentences_conv_activation.get_shape()[2].value], dtype=tf.float32, initializer=tf.constant(0.1))
        x_sentences_dense_activation = tf.tanh(tf.matmul(x_sentences_conv_activation, weights) + bias, name="dense_layer")
        
    #Softmax layer
    y_classified = None
    with tf.variable_scope("softmax_layer") as scope:
        #2 es el número de clases.
        weights = tf.get_variable("softmax_weights", shape=[None,x_sentences_dense_activation.get_shape()[2].value,2])
        bias = tf.get_variable("softmas_bias", shape=[2], initializer=tf.constant(0.1))
        y_logits = tf.matmul(x_sentences_dense_activation, weights) + bias
        y_classified = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_labels, logits=y_logits, name="softmax")
        
    #Loss function
    f_loss = tf.reduce_mean(y_classified, name="loss_calculation")
    train_step = tf.train.AdadeltaOptimizer().minimize(f_loss)
    
    accuracy = None
    with tf.variable_scope("accuracy"):
        prediction_labels = tf.arg_max(y_classified, 1, name="prediction_labels")
        correct_predictions = tf.equal(prediction_labels, y_labels, name="correct_predicionts")
        accuracy = tf.reduce_mean(correct_predictions, name="accuracy")


def get_features(training_sentences, vocabulary):
    """Get the features of the batch_sentences.
    
    Args:
        batch_sentences: Lista de lista de oraciones y palabras.
        vocabulary: Diccionario con las palabras de entrenameinto y su identificador o índice.
    Returns:
        Lista de listas de enteros
     
    """
    sentences_features = []
    own_lower = str.lower
    for sent in training_sentences:
        sentence_features = [vocabulary.get(own_lower(word), OOV_INDEX) for word in sent]
        sentences_features.append(sentence_features)
    return sentences_features


def padding_truncate(training_sentences):
    """Amplia o recorta las oraciones de entrenamiento.
    
    Args:
        training_sentences: Lista de listas de enteros.
    """
    
    for i in range(len(training_sentences)):
        sent_size = len(training_sentences[i])
        if sent_size > MAX_LENGTH:
            training_sentences[i] = training_sentences[i][:MAX_LENGTH]
        elif sent_size < MAX_LENGTH:
            training_sentences[i] += [0] * (MAX_LENGTH - sent_size)
    
    return training_sentences 
            

def model_training(training_sents, training_labels, vocabulary):
    """Entrenamiento del modelo.
    
    Args:
        training_corpus: lista de lista de oraciones
        training_labels:  lista de enteros que se corresponden con las etiquetas
    """
    #Preparación de la entrada: cálculo de características
    training_sents_features = get_features(training_sents, vocabulary)
    #Preparación de la entrada: Ampliación (padding) o truncado (truncate)
    training_sents_features = padding_truncate(training_sents_features)
    
    nn_model = tf.Session()
    
    #Es muy importante este paso, dado que inicializa todas las variables
    nn_model.run(tf.initialize_all_variables())
    
    number_of_batches = int(len(training_sents_features)/SIZE_BATCHES)
    
    for epoch in range(EPOCHS):
        for batch in range(number_of_batches):
             start_index = batch * SIZE_BATCHES
             end_index = (batch + 1) * SIZE_BATCHES
             batch_sentences = training_sents_features[start_index:end_index]
             batch_labels = training_labels[start_index:end_index]
             n_batch_sentences = len(batch_sentences)
             if n_batch_sentences != 0: #Si el tamaño del batch es cero, no se hace nada.
                 #Si el nº. de oracione en el batch es menor que el tamaño del batch, rellenamos con las del principio del corpus de entrenamiento.
                 if n_batch_sentences < SIZE_BATCHES:
                     batch_sentences += training_sents_features[0:(SIZE_BATCHES - n_batch_sentences)]
                     batch_labels += training_labels[0:(SIZE_BATCHES - n_batch_sentences)]

if __name__ == '__main__':
    
    #Definir semilla aleatoria
    random.seed(RANDOM_SEED)
    
    #2.- Leer corpus y partición de entrenamiento y test.
    train_sents, test_sents, train_labels, test_labels = data_preparation_train_test()
    
    #3.- Creación del vocabulario de entrenamiento. Toda palabra que no esté en
    #el vocabulario de entrenamiento se consdierá palabra fuera de vocabulario (00).
    #Si tratamos de asimilarlo a otro problema de aprendizaje automática, las OOV
    #serían datos perdidos.
    
    #¿Por qué se define el inicio de índice en 2? Por que se suele reservar el
    #índice 0 para las palabras 00V, y el índice 1 para el padding (extensión de la entrada de la red)..    
    train_vocabulary = build_vocabulary(train_sents, 2)
    
    #4.- Compilamos el grafo.
    nn_graph()
    
    #5.- Entrenamiento
    
    
    
    