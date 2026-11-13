import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

CARPETA = Path(__file__).parent / "model"

PUERTOS = {"C": "Cherburgo", "Q": "Queenstown", "S": "Southampton"}
SEXOS = {"female": "Mujer", "male": "Hombre"}

st.set_page_config(page_title="Supervivencia en el Titanic", layout="centered")


@st.cache_resource
def cargar_modelo():
    return joblib.load(CARPETA / "titanic_rf.joblib")


@st.cache_data
def cargar_limites():
    return json.loads((CARPETA / "limites.json").read_text(encoding="utf-8"))


modelo = cargar_modelo()
lim = cargar_limites()

st.title("Supervivencia en el Titanic")
st.caption(
    "Ejercicio de clase sobre datos históricos de 1912. Un random forest estima la probabilidad "
    "de sobrevivir de un pasajero a partir de sus datos de embarque. No sirve para predecir nada real."
)

st.subheader("Datos del pasajero")
col1, col2 = st.columns(2)

with col1:
    clase = st.selectbox("Clase del billete", lim["Pclass"], format_func=lambda c: f"{c}ª clase")
    sexo = st.selectbox("Sexo", lim["Sex"], format_func=lambda s: SEXOS.get(s, s))
    puerto = st.selectbox("Puerto de embarque", lim["Embarked"],
                          index=lim["Embarked"].index("S") if "S" in lim["Embarked"] else 0,
                          format_func=lambda p: PUERTOS.get(p, p))

with col2:
    edad_desconocida = st.checkbox("Edad desconocida")
    edad = st.number_input("Edad (años)", min_value=0.0, max_value=120.0,
                           value=float(round(lim["Age"]["mediana"])), step=1.0,
                           disabled=edad_desconocida)
    hermanos = st.number_input("Hermanos o cónyuge a bordo", min_value=0, max_value=15,
                               value=int(lim["SibSp"]["mediana"]), step=1)
    padres = st.number_input("Padres o hijos a bordo", min_value=0, max_value=15,
                             value=int(lim["Parch"]["mediana"]), step=1)

tarifa = st.number_input("Tarifa pagada (libras de 1912)", min_value=0.0, max_value=1000.0,
                         value=float(round(lim["Fare"]["mediana"], 2)), step=1.0)

# Avisos cuando un valor cae fuera de lo que el modelo vio al entrenar.
avisos = []
if not edad_desconocida and not (lim["Age"]["min"] <= edad <= lim["Age"]["max"]):
    avisos.append(f"La edad está fuera del rango de entrenamiento ({lim['Age']['min']:.2f} a {lim['Age']['max']:.0f} años).")
for nombre, valor, clave in [("Hermanos o cónyuge", hermanos, "SibSp"),
                             ("Padres o hijos", padres, "Parch"),
                             ("La tarifa", tarifa, "Fare")]:
    if valor > lim[clave]["max"]:
        avisos.append(f"{nombre}: {valor:g} supera el máximo visto en entrenamiento ({lim[clave]['max']:g}).")
for aviso in avisos:
    st.warning(aviso + " La estimación es menos fiable.")

pasajero = pd.DataFrame([{
    "Age": np.nan if edad_desconocida else edad,
    "SibSp": hermanos,
    "Parch": padres,
    "Fare": tarifa,
    "Pclass": int(clase),
    "Sex": sexo,
    "Embarked": puerto,
}])

p_sobrevive = float(modelo.predict_proba(pasajero)[0, 1])

st.subheader("Estimación del modelo")
m1, m2 = st.columns(2)
m1.metric("Probabilidad de sobrevivir", f"{p_sobrevive:.1%}")
m2.metric("Probabilidad de no sobrevivir", f"{1 - p_sobrevive:.1%}")
st.progress(p_sobrevive, text="Sobrevive")
st.progress(1 - p_sobrevive, text="No sobrevive")
st.caption(f"En el conjunto de entrenamiento sobrevivió el {lim['supervivencia_entrenamiento']:.1%} de los pasajeros.")

with st.expander("Sobre el modelo y sus límites"):
    st.markdown(
        "- Random forest con profundidad máxima 3, elegido con validación cruzada de 5 folds sobre 712 pasajeros.\n"
        "- En una prueba de 179 pasajeros apartados desde el inicio: accuracy 0,788 (línea base 0,615), F1 0,694 y AUC 0,830.\n"
        "- Detecta el 62 % de los supervivientes reales y, cuando predice supervivencia, acierta el 78 % de las veces.\n"
        "- Con solo 179 pasajeros de prueba, estas cifras tienen un margen de error amplio.\n"
        "- Los datos son de un único barco en 1912, así que el modelo describe patrones históricos y no generaliza a otras situaciones."
    )
