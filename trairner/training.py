import tensorflow as tf

# Paso 1: Preparación de los datos

# Aquí debes cargar y preprocesar tus datos según tus necesidades.
# Asegúrate de dividir los datos en conjuntos de entrenamiento, validación y prueba.

# Paso 2: Cargar imagenes para el entrenamiento

train_images =  # Aquí debes cargar las imágenes de entrenamiento
train_labels =  # Aquí debes cargar las etiquetas de entrenamiento
val_images =  # Aquí debes cargar las imágenes de validación
val_labels =  # Aquí debes cargar las etiquetas de validación
    

# Paso 3: Construcción del modelo

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Paso 3: Entrenamiento del modelo

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels))

# Paso 6: Guardar el modelo

model.save('model.h5')

# Paso 5: Evaluación del modelo

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Precisión en el conjunto de prueba:', test_acc)

# Paso 4: Uso del modelo

predictions = model.predict(test_images)


# Path: trairner\training.py
import tensorflow as tf

# Paso 1: Preparación de los datos

# Aquí debes cargar y preprocesar tus datos según tus necesidades.
# Asegúrate de dividir los datos en conjuntos de entrenamiento, validación y prueba.






