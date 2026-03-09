# 📈 Consumer Price Index (CPI) Prediction Engine

### 📝 Purpose & Requirement
The objective of this project is to develop a **Multiple Linear Regression model** capable of predicting the **General Index (CPI)** by analyzing various commodity sub-indices like "Cereals and products," "Meat and fish," and "Fuel and light".

**Why this matters for GenAI:**
This regression model acts as a "Deterministic Tool" for an AI Agent. While an LLM is excellent at explaining economic trends, it can struggle with precise calculations. By integrating this model into a **LangGraph** or **LangChain** workflow, the Agent can perform accurate "Lookups" to provide data-backed answers to user queries.

---

### 🗺️ Project Flow: From Data to Prediction

#### 1. The Plan (Data Engineering)
Before training, the raw data required significant "factory-level" cleaning to resolve `ValueErrors` and `DateParseErrors`:
* **Temporal Alignment**: Merged `Year` and `Month` columns using strict formatting (`%Y-%B`) to ensure the timeline was sorted chronologically rather than alphabetically.
* **Type Enforcement**: Used `pd.to_numeric(errors='coerce')` to strip hidden strings or special characters that would break the mathematical matrix multiplication required for regression.
* **The "Safety Lock"**: Implemented `index.intersection()` to ensure that if a row was dropped in the features ($X$), it was also removed from the target ($y$), preventing "Index De-synchronization".



#### 2. Validation (Measuring Success)
A model is only as good as its performance on unseen data. We evaluate this using two key metrics:
* **Mean Squared Error (MSE)**: Quantifies the average squared distance between our predicted price and the actual price. We aim for the lowest possible value.
* **R-Squared ($R^2$) Score**: Tells us how much of the "General Index" movement is actually explained by our model. A score of 0.95, for example, means our features explain 95% of the variance.



#### 3. Export & Portability
To move this from a Jupyter Notebook to a real-world application, we "serialize" the model:
* **Serialization**: Using `joblib`, we save the trained model as a `.pkl` file. This preserves the "learned weights" so we don't have to retrain every time.
* **The "Agent" API**: The `.pkl` file is loaded into a **FastAPI** or **Flask** wrapper. This turns the model into a URL endpoint that an AI Agent can call whenever it needs a price prediction.

---

### ⚠️ Critical Lessons for Recall
* **Feature Hygiene**: If `X.info()` shows a column as an `object`, the model **will** fail. Regression only speaks the language of `float` and `int`.
* **Data Voids**: Models cannot compute `NaN` (Not a Number). Always decide whether to `fillna(0)` (assume zero) or `dropna()` (remove the record).
* **Target Alignment**: Never clean $X$ without also cleaning $y$. They must remain "Mirror Images" in terms of row count.
