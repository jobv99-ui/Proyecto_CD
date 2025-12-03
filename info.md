# Crear ambiente virtual

conda create -n streamlit_pyspark python=3.10 -y

# Activar el ambiente

conda activate streamlit_pyspark

# Desactivar  el ambiente

conda deactivate

# Si tienes error de java

https://adoptium.net/es/download

# Ejemplo de ejecución 
streamlit run app1.py

# matar o terminar porceso

Control + c

# Instalar dependencias

pip install streamlit pyspark pandas plotly scikit-learn openpyxl

```

### Estructura del Proyecto

proyecto_CD/
│
├── app1.py # Aplicación principal de Streamlit
├── tesla_deliveries_dataset1.csv # Dataset de ventas de autos
├── info.md
├── logo.jpeg # logo

