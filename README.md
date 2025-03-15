# 🚀 NER Model for Recipes: Structuring Cooking Data with AI! 🍕🥑🔥

## 📖 Overview
This project applies **Named Entity Recognition (NER)** to extract structured insights from unstructured **recipe text**. The goal is to identify key entities such as **ingredients, cooking techniques, measurements, and kitchen tools**, enabling applications like **automated grocery list generation, intelligent recipe recommendations, and precise nutritional analysis.**

## 📌 Methodology
- The **RecipeNLG** dataset ([Kaggle Link](https://www.kaggle.com/datasets/)) with **100,000 recipes** was used as the primary data source.  
- A **BERT-based token classification model** was fine-tuned using **Hugging Face Transformers** to recognize **9 predefined entity categories**:  
  - **FRUITS_VEGGIES**  
  - **MEAT_SEAFOOD**  
  - **DAIRY**  
  - **SEEDS_NUTS**  
  - **SEASONINGS**  
  - **COOKING_METHOD**  
  - **COOKING_TOOL**  
  - **MEASURE**  
  - **GRAINS_STARCH**  
- Data preprocessing included **tokenization, label alignment, and entity annotation** using the **BIO tagging scheme**.  
- The model was trained with **optimized hyperparameters**, ensuring robust generalization to unseen recipe text.  

## ⚡ Challenges
- Ensuring proper alignment of multi-word ingredients and subword tokens.  
- Handling sequence length variations without compromising label accuracy.  
- Optimizing batch processing to prevent inconsistencies in training.  

## 📊 Results & Applications
- Achieved an **F1-score of 0.98**, demonstrating high accuracy in detecting named entities.  
- The structured extraction process enables **efficient text search, personalized recipe recommendations, and ingredient-based filtering.**  

## 🚀 How to Use
### 🔹 Install Dependencies
```bash
pip install transformers datasets torch nltk
```
### 🔹 Run Inference
```python
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

# Load Model & Tokenizer
model_path = "bert-ner-finetuned"
model = AutoModelForTokenClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Example Text
def predict_ner(text):
    tokens = tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokens)
    predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()
    return predictions

text = "Chop the onions and mix with cheddar cheese."
print(predict_ner(text))
```

