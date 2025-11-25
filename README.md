# ia1-Matrix_3.0-Es-Demente-
![Imagen de Portada](images/Portada-IA.png)


### **🧠 Exploración del Dataset de Alzheimer (EDA)**

---

### **Descripción del Dataset**

[![Kaggle](https://img.shields.io/badge/Kaggle%20-%20Alzheimer%20Disease%20Dataset-blue?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/ashrafulhossenakash/alzheimer-disease-dataset/data)
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-blue?logo=kaggle)](https://www.kaggle.com/datasets/ashrafulhossenakash/alzheimer-disease-dataset/data)
[![Notebook](https://img.shields.io/badge/-Notebook-blue?logo=jupyter&logoColor=white&labelColor=black)](https://drive.google.com/drive/folders/1S3Z6ylZIzYp-q8UdAKwALf9NoSGyRB02?usp=drive_link)
[![Video Explicativo](https://img.shields.io/badge/Video%20Explicativo-red?logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=sONk6AcXTFw&feature=youtu.be)



## 📑 Índice del Proyecto

1. 🔍 **Primera Fase — Pre-Análisis Conceptual**  
   &nbsp;&nbsp;&nbsp;&nbsp;📘 [Notebook: Primera Entrega](https://colab.research.google.com/drive/1N81qKQA5Ofw4HkcyCfOZbRbB5-XMNCf-?usp=drive_link)  
   &nbsp;&nbsp;&nbsp;&nbsp;🔗 [Ir a la sección](#pre-análisis-conceptual)

2. 🧠 **Aprendizaje Supervisado — Clasificación de Alzheimer (CNN + Transfer Learning)**  
   &nbsp;&nbsp;&nbsp;&nbsp;📘 [Notebook: Segunda Entrega](https://colab.research.google.com/drive/1t11i96hySYd7J6g0pfIePtLrVUIcg2CT?usp=drive_link)  
   &nbsp;&nbsp;&nbsp;&nbsp;🔗 [Ir a la sección](#clasificación-de-alzheimer-con-cnn--transfer-learning)

3. 🧩 **Aprendizaje No Supervisado — PCA y Modelos No Supervisados**  
   &nbsp;&nbsp;&nbsp;&nbsp;📘 [Notebook: Tercera Entrega](https://colab.research.google.com/drive/1ZUsee2ZCrENj-HDeFxn0vh2CmeV3LXFz?usp=sharing)  
   &nbsp;&nbsp;&nbsp;&nbsp;🔗 [Ir a la sección](#pca-modelos-no-supervisados)

4. 📊 **Adicional — PCA + Modelos Supervisados**  
   &nbsp;&nbsp;&nbsp;&nbsp;📘 [Notebook](https://colab.research.google.com/drive/1hvgj8Dl4pi0Q592MtHYm5Z6BiASbOpVo?usp=sharing)  
   &nbsp;&nbsp;&nbsp;&nbsp;🔗 [Ir a la sección](#clasificación-de-alzheimer-con-pca-y-modelos-supervisados)


**Contenido:** Alrededor de 34 mil imagenes de resonancias magnéticas cerebrales, clasificadas en cuatro etapas de la enfermedad: **NonDemented, VeryMildDemented, MildDemented y ModerateDemented.** y en diferente estados: train(entrenamiento), pruebas y val (valoracion).

---

### **Pre-Análisis Conceptual**💡

#### **El Problema: Detección Temprana del Alzheimer**

El Alzheimer es una enfermedad neurodegenerativa que impacta severamente la calidad de vida. 😞 Su diagnóstico temprano es crucial para iniciar tratamientos a tiempo y ofrecer un mejor soporte. La detección tradicional es compleja, por lo que usar resonancias magnéticas y modelos de IA ofrece una vía prometedora para un diagnóstico rápido, estandarizado y automatizado. Esta aproximación podría ser un apoyo valioso para los especialistas.

#### **Objetivo del Análisis (EDA)** 🎯

Nuestro objetivo es entender la estructura y las características de este dataset. ¿Está balanceado? ¿Qué calidad tienen las imágenes? Este análisis exploratorio nos ayudará a identificar desbalances y desafíos inherentes a los datos, lo que es clave para diseñar una estrategia de preprocesamiento robusta. Un EDA bien hecho es el primer paso para entrenar un modelo de clasificación eficaz.

#### **Métricas Clave** 📊

Para evaluar el rendimiento del modelo, usaremos:

* **Accuracy:** Para una visión global del acierto.
* **Precision, Recall y F1-score:** Cruciales para entender cómo el modelo se comporta con cada clase, especialmente las menos representadas.
* **Matriz de Confusión:** Nos permitirá visualizar los errores de clasificación y entender qué clases se confunden más.
* **AUC-ROC:** Para medir la capacidad del modelo para distinguir entre las diferentes etapas de la enfermedad.

Estas métricas son vitales para asegurar que nuestro modelo no solo sea preciso, sino también confiable.

#### **Nuestra Motivación** 💖

El impacto social y clínico del Alzheimer es inmenso. Queremos aplicar la inteligencia artificial a un problema del mundo real con el potencial de generar valor significativo. El reto de usar la visión por computadora para la detección temprana de esta enfermedad nos permite combinar tecnología de vanguardia con un propósito humanitario.

---

### **Post-Análisis (Basado en Datos)** 🔍

#### **Datos Utilizados**

Trabajamos con imágenes de resonancia magnética cerebral, provenientes de Kaggle, organizadas en carpetas para entrenamiento, validación y prueba. El dataset incluye tanto imágenes originales como versiones "aumentadas" (generadas con rotaciones y cambios de contraste) para mejorar la robustez del modelo.

#### **Contenido del Dataset** 🖼️

Las imágenes son cortes axiales del cerebro en escala de grises, etiquetadas como **NonDemented, VeryMildDemented, MildDemented y ModerateDemented**. La inclusión de imágenes aumentadas es un factor clave, ya que introduce variaciones útiles y evita el sobreajuste. La división en subconjuntos (train, val, test) nos permite realizar un entrenamiento y una evaluación justos, asegurando que el modelo se pruebe con datos que nunca ha visto.

**Distribucion por clases**
![Distribucion por clases](images/D_Class.png)

**Resonancias de Ejemplo**
![Resonancias de Ejemplo](images/IMG_Referencia.png)

#### **Desafíos Identificados** ⚠️

Durante el EDA, detectamos varios retos a considerar:

* **Desbalance de Clases:** La cantidad de imágenes no es uniforme en todas las categorías, lo que podría sesgar el modelo.
* **Ruido en las Imágenes Aumentadas:** Aunque útiles, estas imágenes podrían introducir variaciones irrelevantes.
* **Tamaño del Dataset:** Aunque grande para un proyecto académico, puede ser limitado para entrenar modelos de *deep learning* de gran escala.
* **Limitación Clínica:** El dataset se enfoca únicamente en las resonancias, sin incluir otros datos clínicos vitales (edad, género, etc.).


---

[**🔼Volver al inicio**](#ia1-matrix_30-es-demente-)


# 🧠Clasificación de Alzheimer con CNN + Transfer Learning

Este proyecto implementa un modelo de **red neuronal convolucional (CNN)** utilizando **EfficientNetB0** con *transfer learning* para la clasificación de imágenes de resonancias cerebrales en cuatro categorías relacionadas con el Alzheimer:

- **NonDemented**
- **VeryMildDemented**
- **MildDemented**
- **ModerateDemented**

---

## 📌 Paso a paso del proyecto

### 1. Descarga del dataset
El primer paso fue obtener el dataset desde Kaggle. Este conjunto contiene imágenes de resonancias cerebrales organizadas en carpetas para entrenamiento, validación y prueba.  
Esto asegura que el modelo se entrene con datos distintos a los que se usarán para evaluar su desempeño.

```python
import kagglehub

# Descargar dataset desde Kaggle
path = kagglehub.dataset_download("ashrafulhossenakash/alzheimer-disease-dataset")
print("Dataset descargado en:", path)
```

---

### 2. Exploración del dataset
Antes de entrenar, se verificó la distribución de imágenes por clase y conjunto.  
Este análisis inicial permite identificar posibles desbalances de clases y entender la estructura del dataset.

```python
import os

train_path = os.path.join(path, "Alzheimer_Dataset_V2/train")
classes = sorted(os.listdir(train_path))
print("Clases detectadas:", classes)
```
### 📊Distribucion de Clases
![Distribucion de Clases](images/distribucion_class.png)

---

### 3. Configuración de parámetros globales
Se definieron parámetros clave como el tamaño de las imágenes, número de épocas, tasa de aprendizaje y nombres de las clases.  
Estos valores controlan el entrenamiento y permiten reproducir los resultados.

```python
IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 224, 224, 3
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001
NUM_CLASSES = 4
CLASS_NAMES = ['MildDemented','ModerateDemented','NonDemented','VeryMildDemented']
```

---

### 4. Generadores de datos con Data Augmentation
Para mejorar la robustez del modelo, se aplicaron técnicas de *data augmentation*.  
Esto introduce variaciones en las imágenes (rotaciones, zoom, flips) y ayuda a evitar el sobreajuste.

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
```

---

### 5. Construcción del modelo CNN (Transfer Learning)
Se utilizó **EfficientNetB0** como base preentrenada en ImageNet.  
Las capas iniciales se congelaron para preservar el conocimiento previo y se añadieron capas densas personalizadas para la clasificación en 4 categorías.

```python
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
)
base_model.trainable = False  # Fase 1: congelar capas

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'), BatchNormalization(), Dropout(0.5),
    Dense(256, activation='relu'), BatchNormalization(), Dropout(0.4),
    Dense(128, activation='relu'), BatchNormalization(), Dropout(0.3),
    Dense(NUM_CLASSES, activation='softmax')
])
```

---

### 6. Compilación del modelo
Se definió el optimizador **Adam**, la función de pérdida *categorical crossentropy* y la métrica principal *accuracy*.  
Esto permite entrenar el modelo de manera eficiente y evaluar su desempeño.

```python
from tensorflow.keras.optimizers import Adam

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
```

---

### 7. Callbacks configurados
Se añadieron callbacks para mejorar el entrenamiento:  
- **ModelCheckpoint**: guarda el mejor modelo.  
- **EarlyStopping**: detiene el entrenamiento si no hay mejoras.  
- **ReduceLROnPlateau**: ajusta la tasa de aprendizaje automáticamente.

```python
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

checkpoint = ModelCheckpoint("best_model.h5", monitor="val_accuracy", save_best_only=True)
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.5)
```

---

### 8. Entrenamiento del modelo
Finalmente, se entrenó el modelo utilizando los generadores de datos y los *class weights* calculados para compensar ligeros desbalances.  
El historial de entrenamiento se almacenó para luego graficar las curvas de *loss* y *accuracy*.

```python
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop, reduce_lr],
    class_weight=class_weights_dict
)
```



---

## 📊 Resultados esperados

- **Distribución de clases**
      ![**Distribución de clases**](images/distribucion_class.png)

- **Ejemplos de imágenes del dataset**
  - *Imagenes Predichas Correctas* 
      ![Imagenes Predichas Correctas](images/Predicciones-Correctas.png)
  - *Imagenes Predichas Incorrectas*
      ![Imagenes Predichas Incorrectas](images/Predicciones_Incorrectas.png)

- **Curvas de entrenamiento**
      ![**Curvas de entrenamiento**](images/Curvas-ROC.png)
---

[**🔼 Volver al inicio**](#ia1-matrix_30-es-demente-)

# PCA Modelos No Supervisados

## 🧩 Explicación Paso a Paso

### 1. **Carga y Preprocesamiento**
- Se descargan las imágenes del dataset de Alzheimer.  
- Se convierten a escala de grises y se redimensionan a 64x64 píxeles para reducir dimensionalidad.  
- Luego se aplanan en vectores de 4096 características y se normalizan con `StandardScaler`.  

```python
X_flat = X_images.reshape(X_images.shape[0], -1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_flat)
```

👉 **Explicación**: este paso asegura que todas las características tengan la misma escala, lo cual es crítico para PCA y clustering.

---

### 2. **Reducción de Dimensionalidad con PCA**
- PCA se usa para comprimir la información en menos dimensiones manteniendo la mayor varianza posible.  
- Se seleccionaron 50 componentes principales, que explican ≈80% de la varianza.  

```python
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_scaled)
```

👉 **Explicación**: PCA facilita la visualización y mejora la eficiencia de los algoritmos de clustering al eliminar redundancia.

---

### 3. **Clustering con Diferentes Algoritmos**

#### 🔹 K-Means
- Busca particionar los datos en *k* clusters esféricos.  
- Se selecciona *k* óptimo con el método del codo y Silhouette.  

```python
kmeans = KMeans(n_clusters=4, random_state=42)
labels_kmeans = kmeans.fit_predict(X_pca)
```

👉 **Explicación**: K-Means es rápido y genera clusters compactos, pero asume formas esféricas y tamaños similares.

---

#### 🔹 DBSCAN
- Agrupa puntos densos y marca outliers como ruido.  
- Se ajusta el parámetro `eps` con el gráfico de k-distancia.  

```python
dbscan = DBSCAN(eps=5, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_pca)
```

👉 **Explicación**: DBSCAN detecta clusters de formas arbitrarias y maneja bien ruido, lo que lo hace más realista en datos médicos.

---

#### 🔹 Agglomerative Clustering
- Construye una jerarquía de clusters usando el criterio de enlace Ward.  
- Se corta el dendrograma para obtener el número óptimo de clusters.  

```python
agg = AgglomerativeClustering(n_clusters=4, linkage='ward')
labels_agg = agg.fit_predict(X_pca)
```

👉 **Explicación**: este método captura estructuras jerárquicas y anidadas, útil para analizar relaciones entre subgrupos.

---

### 4. **Evaluación con Etiquetas Reales**
- Se comparan los clusters con las clases clínicas usando métricas externas: Accuracy, Precision, Recall, F1.  

```python
def evaluar(labels_pred, labels_true):
    return {
        "Accuracy": accuracy_score(labels_true, labels_pred),
        "Precision": precision_score(labels_true, labels_pred, average='weighted'),
        "Recall": recall_score(labels_true, labels_pred, average='weighted'),
        "F1": f1_score(labels_true, labels_pred, average='weighted')
    }
```

👉 **Explicación**: estas métricas permiten medir qué tan bien los clusters reflejan las etiquetas médicas reales.

**K_Means**
![**K_Means**](images/PCA_NS_K.png)

**DBSCAN Y AGGLOMERATIVE**
![**DBSCAN**](images/PCA_NS_DA.png)


---

## 📊 Resultados
- **K-Means y Agglomerative**: Silhouette ≈ 0.45–0.47, clusters compactos y geométricamente claros.  
- **DBSCAN**: mejor alineación con etiquetas clínicas (Accuracy ≈ 0.46, Precision ≈ 0.59, F1 ≈ 0.42), aunque menos cohesión interna.  
- **Interpretación**: los métodos basados en centroides generan clusters “bonitos”, pero DBSCAN captura mejor la estructura real de las clases médicas.

---

## ✅ Conclusión Final
- **PCA** fue esencial para reducir dimensionalidad y facilitar clustering.  
- **K-Means y Agglomerative** → mejor cohesión interna.

  **METRICAS INTERNAS**
  ![METRICAS INTERNAS](images/PCA_MI.png)
  
- **DBSCAN** → más fiel a la distribución clínica real.

  **METRICAS EXTERNAS**
  ![METRICAS EXTERNAS](images/PCA_ME.png)
    
- En este contexto, **DBSCAN es el más alineado con las etiquetas médicas**, aunque sacrifica cohesión interna.


---

[**🔼 Volver al inicio**](#ia1-matrix_30-es-demente-)


## 🧩 Reduccion de Dimensionalidad

### 2. **Reducción de Dimensionalidad con PCA**
- Se aplicó **PCA** sobre las imágenes aplanadas y normalizadas.  
- Se probaron diferentes configuraciones: 2D, 3D y 50 componentes principales.  
- Se analizó la varianza explicada para cada caso.

```python
from sklearn.decomposition import PCA

# PCA en 2D
pca_2d = PCA(n_components=2, random_state=42)
X_train_pca_2d = pca_2d.fit_transform(X_train_scaled)
X_test_pca_2d = pca_2d.transform(X_test_scaled)

# PCA en 3D
pca_3d = PCA(n_components=3, random_state=42)
X_train_pca_3d = pca_3d.fit_transform(X_train_scaled)
X_test_pca_3d = pca_3d.transform(X_test_scaled)

# PCA con 50 componentes
pca_50 = PCA(n_components=50, random_state=42)
X_train_pca_50 = pca_50.fit_transform(X_train_scaled)
X_test_pca_50 = pca_50.transform(X_test_scaled)

print("Varianza explicada 2D:", sum(pca_2d.explained_variance_ratio_))
print("Varianza explicada 3D:", sum(pca_3d.explained_variance_ratio_))
print("Varianza explicada 50D:", sum(pca_50.explained_variance_ratio_))
```

👉 **Explicación**:  
- PCA 2D y 3D permiten visualizar la distribución de las clases.  
- PCA con 50 componentes conserva ≈80% de la varianza, lo que reduce dimensionalidad sin perder demasiada información.  

---

### 3. **Visualización con PCA**
Se graficaron los datos reducidos para observar la separación de clases.

```python
import matplotlib.pyplot as plt

plt.scatter(X_train_pca_2d[:,0], X_train_pca_2d[:,1], c=y_train, cmap='viridis', alpha=0.6)
plt.title("Visualización PCA 2D")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
```
**Visualizacion PCA 2D**
![visualizacion2DPCA](images/RPCA_2D.png)

**Visualizacion PCA 3D**
![visualizacion3DPCA](images/RPCA_3D.png)

👉 **Explicación**:  
Las clases muestran cierta separación, aunque con solapamientos. PCA captura patrones globales pero no garantiza separación perfecta.

---

### 4. **Visualización con T-SNE**
Se aplicó **T-SNE** para explorar relaciones no lineales y visualizar agrupamientos más definidos.

```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_train_tsne = tsne.fit_transform(X_train_scaled)

plt.figure(figsize=(10,6))
plt.scatter(X_train_tsne[:,0], X_train_tsne[:,1], c=y_train, cmap='viridis', alpha=0.6, s=20)
plt.title("Visualización T-SNE")
plt.xlabel("TSNE-1")
plt.ylabel("TSNE-2")
plt.show()
```

👉 **Explicación**:  
- T-SNE preserva **distancias locales**, mostrando agrupamientos más claros que PCA.  
- Es sensible a parámetros como `perplexity` y `n_iter`.  
- Se usa principalmente para **visualización exploratoria**, no para clustering directo.

**Visualizacion T-SNE 2D**
![visualizacion2DTNSE](images/RTNSE_2D.png)

**Visualizacion T-SNE 3D**
![visualizacion2DTNSE](images/RTNSE_3D.png)

---

### 5. **Clustering sobre PCA y T-SNE**
Se aplicaron algoritmos de clustering sobre las representaciones reducidas:

```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# K-Means sobre PCA 50D
kmeans = KMeans(n_clusters=4, random_state=42)
labels_kmeans = kmeans.fit_predict(X_train_pca_50)

# DBSCAN sobre PCA 50D
dbscan = DBSCAN(eps=5, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_train_pca_50)

# Agglomerative sobre PCA 50D
agg = AgglomerativeClustering(n_clusters=4, linkage='ward')
labels_agg = agg.fit_predict(X_train_pca_50)
```

👉 **Explicación**:  
- PCA + clustering → más eficiente y con clusters compactos.  
- T-SNE → útil para visualizar cómo los clusters se distribuyen, aunque no se usa como entrada directa para clustering por su naturaleza estocástica.  

---

[**🔼 Volver al inicio**](#ia1-matrix_30-es-demente-)

# 🧠Clasificación de Alzheimer con PCA y Modelos Supervisados

Este proyecto aplica **reducción de dimensionalidad (PCA)** y distintos algoritmos de **clasificación supervisada** sobre imágenes de resonancias cerebrales para identificar estados de demencia.

---

## ⚙️ Flujo de trabajo

### 1. **Carga y preparación de datos**
- Se descargó el dataset de Alzheimer desde Kaggle.
- Se cargaron imágenes en escala de grises, redimensionadas a 64x64.
- Se dividieron en conjuntos de entrenamiento y prueba.

```python
# Carga optimizada de imágenes
X_train, y_train = load_images(TRAIN_PATH, CLASSES)
X_test, y_test   = load_images(TEST_PATH, CLASSES)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
```

---

### 2. **Aplanado y reducción de dimensionalidad con PCA**
- Se aplanaron las imágenes (64x64 → 4096 features).
- Se estandarizaron los datos.
- Se aplicó PCA con 50 componentes, reteniendo ~80% de la varianza.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat  = X_test.reshape(X_test.shape[0], -1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flat)
X_test_scaled  = scaler.transform(X_test_flat)

pca = PCA(n_components=50, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)

print(f"Varianza explicada: {sum(pca.explained_variance_ratio_)*100:.2f}%")
```

---

### 3. **Entrenamiento de modelos supervisados**
Se entrenaron tres modelos clásicos y una red neuronal:
- **Random Forest** → mejor desempeño (Accuracy ≈ 0.66, F1 ≈ 0.67).
- **Decision Tree** → desempeño más bajo.
- **SVM (RBF)** → resultados intermedios.
- **Red Neuronal (MLP compacto)** → Accuracy ≈ 0.58.

```python
# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train_pca, y_train)
y_pred_rf = rf.predict(X_test_pca)
```

```python
# Red Neuronal
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

model = Sequential([
    Dense(128, activation='relu', input_shape=(50,)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(4, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

---

### 4. **Evaluación y comparación**
- Se calcularon métricas: Accuracy, F1, Precision, Recall.
- Se generaron **tablas comparativas**, **gráficas de métricas**, **matrices de confusión** y **curvas de pérdida** para la red neuronal.
- El **Random Forest** fue el modelo más robusto en este flujo.

```python
import pandas as pd
df_results = pd.DataFrame(results)
print(df_results.sort_values("F1-Score", ascending=False))
```

**Resultado Evaluacion metricas**
 ![**Resultado Evaluacion metricas**](images/PCA_Metricas.png)


**Matriz de Confusion**
 ![**Matriz de Confusion**](images/PCA_Matriz.png)



Esto deja claro que **Random Forest con PCA** fue la mejor opción en este caso, aunque se exploraron alternativas como SVM y redes neuronales.

---

### **Curso:** Inteligencia Artificial I -2025-2 C1
### **Grupo:** Matrix 3.0
### **Integrantes:**
* Harold Esteban Duran Osma-2225113
* Yeison Steven Ovalle Manjarres-2225115
* David Santiago Sáenz Ortiz-2215506
