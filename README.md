# Supervivencia en el Titanic: app de Streamlit

Clasificador de supervivencia del Titanic (random forest) desplegado como app de Streamlit.

**App en vivo:** https://streamlit-titanic-classifier.onrender.com

## Qué hace

Estima la probabilidad de que un pasajero sobreviviera a partir de su clase, sexo, edad, familiares a bordo, tarifa y puerto de embarque. Los controles toman sus rangos de los datos de entrenamiento y la app avisa cuando un valor queda fuera de ellos.

## Método

- Datos: `train.csv` de Kaggle (891 pasajeros), con partición estratificada 80/20 (712 entrenamiento, 179 prueba, semilla 42).
- Preprocesado en un `Pipeline` de scikit-learn: imputación de edad con indicador de faltante, log de la tarifa, escalado y one-hot de las categóricas.
- Selección por validación cruzada de 5 folds, con reglas fijadas antes de ver resultados: entre cinco candidatos se eligió random forest, y entre 32 combinaciones de hiperparámetros la de menor profundidad dentro de una desviación del mejor F1.
- Modelo final: random forest, 300 árboles, `max_depth=3`, `min_samples_leaf=5`, `max_features=0,5`.

## Resultados

En el conjunto de prueba (179 pasajeros, evaluado una sola vez): accuracy 0,788 (línea base 0,615), F1 0,694, AUC 0,830. Detecta el 62 % de los supervivientes y acierta el 78 % de las veces que predice supervivencia.

## Limitaciones

- La prueba tiene solo 69 supervivientes, así que las cifras tienen un margen de error amplio.
- El ajuste de hiperparámetros no mejoró de forma demostrable al modelo por defecto (diferencias dentro del ruido entre folds).
- Datos de un único barco en 1912: describe patrones históricos y no sirve para predecir nada real.
- En el plan gratuito de Render la app se duerme tras un rato sin visitas y la primera carga tarda unos segundos.

## Estructura

- `notebook/eda.ipynb`: análisis, selección, evaluación y generación de los archivos de la app.
- `src/app.py`: app de Streamlit. `src/model/`: modelo y límites. `src/requirements.txt`: dependencias con versiones fijadas.
- `models/`: copia del modelo guardado desde el notebook.

## Ejecutar en local

    pip install -r src/requirements.txt
    streamlit run src/app.py
