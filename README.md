# DAY-25-AM
Assignment — Week 05 · Day 25 (AM Session)
Here’s a clean, submission-ready **README.md** tailored to your repo structure and assignment context:

---

# 📊 DAY-25-AM

**Assignment — Week 05 · Day 25 (AM Session)**

---

## 📌 Overview

This repository contains the work completed for **Day 25 (AM Session)** of the Week 05 assignment. The focus of this assignment is on **data analysis, preprocessing, and issue handling** in real-world datasets.

The project demonstrates how to identify and handle common data quality problems such as:

* Outliers
* Missing values
* Invalid entries
* Structural inconsistencies

---

## 📂 Repository Structure

```
DAY-25-AM/
│
├── LICENSE                # MIT License
├── README.md              # Project documentation
├── Part_D.md              # Conceptual explanations / answers
├── part_A_B.ipynb         # Implementation for Part A & B
├── part_A_B_C.ipynb       # Extended implementation (A, B, C)
```

---

## 🎯 Objectives

* Perform **Exploratory Data Analysis (EDA)**
* Identify **data quality issues**
* Apply appropriate **data cleaning techniques**
* Prepare data for further modeling

---

## ⚙️ Key Data Issues & Handling

### 1. Outliers in `order_value`

* **Problem:** Extreme values skew analysis
* **Solution:** Capping using IQR or percentile method

---

### 2. Invalid `delivery_time` (null/zero)

* **Problem:** Missing or incorrect delivery durations
* **Solution:**

  * Replace with median/mean
  * Or remove invalid rows depending on context

---

### 3. Incorrect `prep_time`

* **Problem:** Data entry errors (negative or unrealistic values)
* **Solution:**

  * Convert to absolute values
  * Apply logical thresholds

---

### 4. Customer Rating Issues (Structural)

* **Problem:** Inconsistent or biased rating distribution
* **Solution:**

  * Normalize or standardize ratings
  * Handle missing/invalid entries

---

## 📊 Techniques Used

* Data Cleaning
* Feature Transformation
* Outlier Treatment
* Handling Missing Values
* Basic EDA

---

## 🛠️ Tools & Libraries

* Python
* Pandas
* NumPy
* Matplotlib / Seaborn

---

## 🚀 How to Run

1. Clone the repository

```bash
git clone <your-repo-link>
cd DAY-25-AM
```

2. Open Jupyter Notebook

```bash
jupyter notebook
```

3. Run:

* `part_A_B.ipynb`
* `part_A_B_C.ipynb`

---

## 📈 Outcome

* Cleaned and structured dataset
* Improved data reliability
* Ready for machine learning or further analysis

---

## 📄 License

This project is licensed under the **MIT License**.

---

## ✍️ Author

**Avishka Jindal**

---

If you want, I can **upgrade this README to a high-scoring version (with visuals, EDA insights, and results section based on your notebook outputs)** — that’s usually what impresses evaluators.
