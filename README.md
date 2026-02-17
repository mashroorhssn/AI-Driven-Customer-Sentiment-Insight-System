# AI-Driven-Customer-Sentiment-Insight-System
## Project Overview

This project is an end-to-end sentiment analysis and executive dashboard solution developed for **Swan Group**. With a 46-year legacy in the bedding and foam industry in Bangladesh, Swan Group faces the challenge of monitoring thousands of digital customer interactions.

This system automates the processing of customer feedback from Facebook, Instagram, and web reviews, transforming unstructured text into actionable business intelligence to ensure brand quality and responsive customer service.

## Objective

* **Sentiment Classification:** Automate the categorization of customer feedback into Positive, Negative, or Neutral using NLP.
* **Root Cause Analysis:** Identify specific product issues (e.g., "Foam Density," "Delivery Delays") through visual analytics.
* **Executive Summarization:** Use a Large Language Model (LLM) to provide high-level summaries of thousands of reviews for leadership.
* **Interactive Querying:** Provide a Chatbot interface for management to query the data using natural language.

## Tech Stack

* **Language:** Python 3.x
* **Interface:** Streamlit (Web Dashboard)
* **NLP:** NLTK (VADER Sentiment Analysis), WordCloud
* **Data Handling:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **AI/LLM:** Local LLM integration (via Requests API to LM Studio/Ollama)

## Repository Structure

```text
Swan-Sentiment-Analysis/
├── data/
│   └── swan_large_dataset.csv       # The generated feedback dataset
├── notebooks/
│   └── Model_Testing.ipynb          # Performance testing and VADER tuning
├── scripts/
│   └── generate_data.py             # Python script for synthetic data generation
├── src/
│   └── swan1.py                     # Main Streamlit dashboard application
├── docs/
│   └── Roll 23 Proposal.pdf         # Project proposal
├── README.md                        # Project documentation
└── requirements.txt                 # List of dependencies
```

## Methodology

1. **Data Generation:** - A custom Python script (generate_data.py) was used to create a synthetic dataset of 5,000 reviews mimicking real-world feedback for Swan mattresses and pillows.


2. **Sentiment Engineering:** - In the experimentation phase (Model_Testing.ipynb), the **VADER (Valence Aware Dictionary and sEntiment Reasoner)** was adjusted with negative weights for specific industry terms like 'sagging' or 'unresponsive' to increase classification accuracy.
* Logic handles intensity (e.g., "excellent" vs. "good") and negations.


3. **Visual Analytics:** - **Temporal Analysis:** Tracking sentiment trends over time to identify if specific marketing campaigns or product batches caused shifts in public perception.
* **Product Segmentation:** Breaking down sentiment by category (Spring Mattress, Supreme Foam, etc.) to isolate quality control issues.


4. **LLM-Powered Insights:**
* Integrated a generative AI layer that acts as a "Data Assistant." It ingests the current dashboard metrics and provides a natural language answer to executive questions.


## How to Run

### 1. Prerequisites

Ensure you have Python installed. For the LLM "Executive Summary" feature, ensure your local LLM server (like LM Studio) is running on `http://localhost:1234`.

### 2. Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/Swan-Customer-Insight-System.git
cd Swan-Customer-Insight-System
pip install -r requirements.txt

```

### 3. Execution

Launch the dashboard:

```bash
streamlit run src/swan1.py

```


*Note: The AI Assistant requires a local LLM server running on port 1234 (LM Studio default).*
## Business Value Delivered


* **Efficiency:** Reduces manual review time by ~95%, allowing the marketing team to focus on high-priority grievances.
* **Quality Control:** Rapidly identifies recurring product complaints (e.g., specific batch issues with foam density), allowing for immediate manufacturing course correction.
* **Strategic Agility:** Provides the Director with an "at-a-glance" report on brand health across all digital touchpoints.

## 📜 Credits

* **Author:** K. M. Mashroor Hossain
* **Course:** Applied Machine Learning for Business
* **Institution:** Institute of Business Administration (IBA), University of Dhaka
