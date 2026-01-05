# ⚠️ SETUP FIRST - Run This on New Laptop

## Step-by-Step Setup

### 1. Create Virtual Environment
```powershell
python -m venv venv
```

### 2. Activate Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install ALL Dependencies (REQUIRED!)
```powershell
pip install -r requirements.txt
```

**This installs:**
- pytest (for testing)
- flake8 (for linting)  
- jupyter + ipykernel (for notebooks)
- All ML libraries (matplotlib, pandas, etc.)

### 4. Verify Installation
```powershell
pytest --version
# Should show: pytest 7.4.3

flake8 --version
# Should show: 6.1.0

python -c "import matplotlib; print('matplotlib OK')"
# Should show: matplotlib OK
```

### 5. Setup Jupyter Kernel (For Notebook)
**Open assignment1.ipynb:**
1. Click the **kernel selector** in top-right corner (shows Python version)
2. Select **"Python Environments"**
3. Choose **your venv interpreter**: `.\venv\Scripts\python.exe`

Now all notebook cells will use your venv with matplotlib installed.

### 6. Generate Model Files (Required for Docker API)
Model `.pkl` files are not in git (too large). Generate them by running the notebook:

```powershell
# Open assignment1.ipynb in VS Code
# Run all cells (Cell → Run All)
# This creates: random_forest_model.pkl, imputer.pkl, logistic_model.pkl
```

Then copy to api folder:
```powershell
Copy-Item random_forest_model.pkl api/models/
Copy-Item imputer.pkl api/models/
```

---

## ❌ Common Errors

**Error:** `pytest : The term 'pytest' is not recognized`  
**Cause:** You didn't run `pip install -r requirements.txt`

**Error:** `No module named 'matplotlib'` in notebook  
**Cause:** Notebook using wrong kernel. Follow step 5 above to select venv kernel.

**Error:** `COPY models/ models/` fails in Docker build  
**Cause:** Model files don't exist. Run notebook to generate them (step 6).

**Solution:**
```powershell
# Make sure venv is activated (you see (venv) in prompt)
pip install -r requirements.txt
```

---

## ✅ After Setup

Now you can run:
- `pytest -v` - Run tests
- `jupyter notebook assignment1.ipynb` - Run notebook
- `flake8 src tests` - Run linting

See README.md for all commands.
