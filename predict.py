# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#instancia de teste = 1,85,66,29,0,26.6,0.351,31,0
#teste=np.array([1,85,66,29,0,26.6,0.351,31])

#instancia de teste = 5,166,72,19,175,25.8,0.587,51,1
teste=np.array([5,166,72,19,175,25.8,0.587,51])

dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
X = dataset[:,0:8]
Y = dataset[:,8]

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, epochs=150, batch_size=10)

teste=np.asmatrix(teste)
result=model.predict(teste)
print(teste,' previsto: ',result[0])
