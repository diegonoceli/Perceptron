import numpy as np
from keras.models import Sequential
from keras.layers import Dense

x=np.array([[1],[2],[3],[4],[5]])
y=np.array([[3],[6],[9],[12],[15]])

model=Sequential()
model.add(Dense(1, input_dim=1))
model.compile(optimizer='sgd', loss='mse', metrics=['acc'])
model.fit(x,y,epochs=2000)

while True:
	i=input('Digite: ')
	t=float(i)
	t=np.asmatrix(t)
	result=model.predict(t)
	print(i,' previsto: ',result[0])