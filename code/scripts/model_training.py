
from keras.utils import to_categorical
from keras import models
from keras import layers

from sklearn.model_selection import train_test_split

input_dim = X.shape[1]

model = models.Sequential()
model.add(layers.Dense(50, activation = "relu", input_shape=(input_dim, )))
# Hidden - Layers
model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation = "relu"))
model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation = "relu"))
# Output- Layer
model.add(layers.Dense(1, activation = "sigmoid"))
model.summary()

data_x = X.toarray()
y = df["score"].values

train_X, X_test, train_y, y_test = train_test_split(data_x, y, test_size = 0.2, shuffle = True, random_state=4)
X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, test_size = 0.2, shuffle = True, random_state=4)


model.compile(
optimizer = "adam",
loss = "binary_crossentropy",
metrics = ["accuracy"]
)

results = model.fit(
X_train, y_train,
epochs= 100,
batch_size = 32,
validation_data = (X_val, y_val)
)