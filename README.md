PS C:\Users\User\Desktop\SuicideWatchAI>
python --version
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip check
python -m flask --version

python .\preprocessing.py
python .\train_models.py
python .\genetic_optimizer.py
python .\train_models.py
python .\genetic_optimizer_v2.py
python .\train_ga_v2_model.py
python .\evaluation.py
python .\app.py

Quick Start — Already Trained Project
python --version
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python .\app.py
click on : http://127.0.0.1:5000