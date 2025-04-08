# Movie Review Sentiment Analysis

This project aims to **train various models to predict the sentiment of movie reviews** (positive or negative).  The dataset consists of 50,000 movie reviews, and the task is a binary classification.

---

## Project Structure

- **cleaning.ipynb**  

  - Handles **data preprocessing and cleaning**.  

  - Loads the raw movie review dataset, performs text cleaning (e.g. converting to lowercase, removing HTML tags, punctuation etc) and saves the cleaned data for modeling.

- **modelling.ipynb**  

  In this notebook, I train and evaluate an LSTM and custom Transformer model using **TensorFlow/Keras**:
  
- **bert_model.ipynb**  

  In this notebook, I Fine-tune a **BERT (Bidirectional Encoder Representations from Transformers)** model for sentiment analysis using **PyTorch** and **HuggingFace Transformers** and evaluate and compare its performance to the other models trained in **modelling.ipynb** 
---

## Setup Instructions

### 1. Running **modelling.ipynb**

- Ensure to use **Python version >3.9** (preferably **Python 3.12.0**).
- Install required libraries by running:

```bash
pip install -r requirements.txt
```
### 2. Running **bert_model.ipynb**

-   **Important** : Use Python 3.10.16 for compatibility

-   This project uses **PyTorch 2.5.1** with **CUDA 12.1**.
To install the correct version of PyTorch run :


```bash
pip install torch==2.5.1+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

```

-   Install other required libraries by running

```bash
pip install -r requirements_bert.txt
```

