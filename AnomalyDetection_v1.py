import numpy as np
from tensorflow.keras.models import Sequential
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.cluster import KMeans
import tensorflow as tf
import time

class KmeansNeuralNet():

    def __init__(self, input_dim, rd , k, filters=7 , kernel_size=9):

        self.input_dim = input_dim
        self.rd = rd
        self.k=k
        self.filters = filters
        self.kernel_size = kernel_size


    def custom_loss(self, w1, w2):

        def custom_hinge(_,y_pred):
            centor = tf.reduce_mean(y_pred, axis=0)
            dist = K.sum((y_pred - centor)**2, axis=1)
            loss = 0.5 * tf.reduce_sum(w1 **2) + 0.5 * tf.reduce_sum(w2 ** 2) +tf.reduce_sum(dist)
            return loss

        return custom_hinge

    def build_model(self):

        input_dim = self.input_dim
        rd = self.rd
        filters = self.filters
        kernel_size = self.kernel_size


        model = Sequential()
        hidden=Conv1D(filters,kernel_size,input_shape=(input_dim,1))
        model.add(hidden)
        model.add(Activation('tanh'))

        pooling = MaxPooling1D(2)
        model.add(pooling)

        model.add(Flatten())

        hidden=Dense(32,activation='relu')
        model.add(hidden)

        hidden_output = Dense(rd)
        model.add(hidden_output)

        w1 = hidden.get_weights()[0]
        w2 = hidden_output.get_weights()[0]

        return [model, w1, w2]


    def train_model(self, x, epochs=100, init_lr=1e-2):

        [model, w1, w2] = self.build_model()

        model.compile(optimizer=Adam(lr=init_lr, decay=init_lr/epochs),
                      loss=self.custom_loss(w1,w2))

        history = model.fit(x, np.zeros((x.shape[0],)),
                            steps_per_epoch=1,
                            shuffle=True,
                            epochs=epochs,
                            )

        kmeans = KMeans(n_clusters=self.k)
        fit_transform = kmeans.fit_transform(model.predict(x))

        return history, model, kmeans, fit_transform

def dist_function (data_center, data):
    dist = np.sqrt(K.sum((data-data_center)**2, axis=1))
    return dist

def detection(kmeans, fit_transform, test, k, r=1):

    start_time=time.time()
    print('start detection!', start_time)

    center = kmeans.cluster_centers_
    center = tf.split(center,k,axis=0)

    distance=np.empty([k,test.shape[0]])
    for i in range(k):
        distance[i]=dist_function(center[i],test)

    labels = np.argmin(distance,axis=0)
    detect = np.empty([k,test.shape[0]])
    for i in range(k):
        detect[i]=[distance[0,j] > np.quantile(fit_transform[:, i],r)for j in range(test.shape[0])]

    result = detect[0]
    for i in range(k-1):
        result = result * detect[i+1]

    end_time = time.time()

    print("WorkingTime: {} sec".format(end_time-start_time))
    return [result,labels]