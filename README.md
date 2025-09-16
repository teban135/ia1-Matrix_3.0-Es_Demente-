# ia1-Matrix_3.0-Es_Demente-


### **🧠 Exploración del Dataset de Alzheimer (EDA)**

---

### **Descripción del Dataset**

**Nombre:** [Alzheimer Disease Dataset](https://www.kaggle.com/datasets/ashrafulhossenakash/alzheimer-disease-dataset/data)

**Fuente:** [Kaggle](https://www.kaggle.com/datasets/ashrafulhossenakash/alzheimer-disease-dataset/data)

🔗[**Enlace Notebook**](https://colab.research.google.com/drive/1N81qKQA5Ofw4HkcyCfOZbRbB5-XMNCf-?usp=drive_link)

**Contenido:** Más de 6,400 imágenes de resonancias magnéticas cerebrales, clasificadas en cuatro etapas de la enfermedad: **NonDemented, VeryMildDemented, MildDemented y ModerateDemented.**

---

### **Pre-Análisis Conceptual** 💡

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

#### **Desafíos Identificados** ⚠️

Durante el EDA, detectamos varios retos a considerar:

* **Desbalance de Clases:** La cantidad de imágenes no es uniforme en todas las categorías, lo que podría sesgar el modelo.
* **Ruido en las Imágenes Aumentadas:** Aunque útiles, estas imágenes podrían introducir variaciones irrelevantes.
* **Tamaño del Dataset:** Aunque grande para un proyecto académico, puede ser limitado para entrenar modelos de *deep learning* de gran escala.
* **Limitación Clínica:** El dataset se enfoca únicamente en las resonancias, sin incluir otros datos clínicos vitales (edad, género, etc.).

### **Curso:** Inteligencia Artificial I -2025-2 C1
### **Grupo:** Matrix 3.0
### **Integrantes:**
* Harold Esteban Duran Osma-2225113
* Yeison Steven Ovalle Manjarres-2225115
* David Santiago Sáenz Ortiz-2215506
