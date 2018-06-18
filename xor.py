import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

x=np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([[0],[1],[1],[0]])

model=Sequential()

model.add(Dense(4, input_dim=2, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))

sgd = SGD(lr=0.1)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(x,y,epochs=1000)

while True:
	i=input('Digite: ')
	j=input('Digite: ')
	t=float(i),float(j)
	t=np.asmatrix(t)
	result=model.predict(t)
	print(i,' previsto: ',result[0])
