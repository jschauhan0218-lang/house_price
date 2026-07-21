import sys
import sklearn
import streamlit

print("Python:", sys.executable)
print("Streamlit:", streamlit.__file__)
print("Scikit-learn:", sklearn.__file__)
print("Scikit-learn version:", sklearn.__version__)