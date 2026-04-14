@echo off

#echo Installing dependencies...
#pip install -r requirements.txt -q
#echo.

echo Running preprocessing pipeline...
python -m src.preprocessing
echo.
echo Training models with MultinomialNB...
python -m src.train_model
echo.
echo Starting app...
streamlit run app.py --server.headless true --server.port 8501


