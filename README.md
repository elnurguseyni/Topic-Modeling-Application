#  Topic Modeling Application

A comprehensive, Streamlit-based application for performing and evaluating topic modeling using both Python and R. Supports classical, embedding-based, and metadata-aware approaches with consistent preprocessing, interactive UI, and GPT-powered labeling.

---

##  Quick-Start Guide

Ensure Python (≥3.9), R (≥4.2), and Git are installed.

### macOS (with Homebrew):
```bash
brew install python r git openblas
```

### Ubuntu/WSL:
```bash
sudo apt update && sudo apt install python3 python3-venv r-base git build-essential libopenblas-dev
```

### Clone the repository and set up Python environment:
```bash
git clone https://github.com/elnurguseyni/Topic-Modeling-Application.git
cd Topic-Modeling-Application

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Install required R packages by running in an R console:
```R
install.packages(c("stm", "tm", "SnowballC", "readr", "dplyr"))
```

---

## ⚙ Quick Commands

```bash
# Activate Python environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run Streamlit unified topic modeling app (LDA, BERTopic, Top2Vec)
streamlit run main_topic_modeling_app.py

# Run STM Streamlit interface (calls STM R script via subprocess)
streamlit run stm_streamlit_interface.py

# (Optional) Manually run STM R script for debugging
# Rscript stm_model_runner.R
```

**Note:**
- `stm_streamlit_interface.py` automatically calls `stm_model_runner.R` using Python’s `subprocess`.
- The main app (`main_topic_modeling_app.py`) directly handles LDA, BERTopic, and Top2Vec in Python.

Now open your browser and go to:  
👉 [http://localhost:8501](http://localhost:8501)

Upload your dataset, configure preprocessing, select a model (LDA, BERTopic, Top2Vec, STM), and run the analysis.  
All results and logs are saved in the `logs/` directory.

### Optional: GPT-generated topic labels  
To enable GPT-based labeling, provide your OpenAI API key via the sidebar or set as an environment variable:
```bash
export OPENAI_API_KEY="your-api-key"
```

---

## 📁 Project Structure

```
├── main_topic_modeling_app.py      # Streamlit frontend for LDA, BERTopic, Top2Vec (Python)
├── stm_streamlit_interface.py      # Streamlit interface for STM (Python, calls R)
├── stm_model_runner.R              # STM modeling backend (R)
├── requirements.txt                # Python dependencies
└── logs/                           # Output logs and topic data
```

---

## 📝 Citation

If you use this app in your research or project, please cite it:

```
@software{Huseynov_2025_TopicApp,
  author  = {E. Huseynov},
  title   = {Topic-Modeling-Application: A Unified Python–R Streamlit Tool},
  year    = 2025,
  url     = {https://github.com/elnurguseyni/Topic-Modeling-Application}
}
```

---

Enjoy your topic modeling! 🚀
