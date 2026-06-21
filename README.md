# CUMCM Team Project: Bike-sharing Demand Forecasting and Dispatch Optimization

## Overview

This project was completed as a team project for the China Undergraduate Mathematical Contest in Modeling. The project focuses on analysing bike-sharing station data, forecasting hourly borrowing and returning demand, and designing a dispatch optimization strategy.

I was mainly responsible for the coding implementation, including data preprocessing, visualization, demand forecasting, model comparison, and dispatch optimization.

## Main Components

### 1. Data Analysis and Visualization

The script `统计图.py` analyses station-level bike-sharing data and generates visualizations, including station borrowing patterns, spatial distribution, and representative station trend comparisons.

### 2. Demand Forecasting

The script `预测模型.py` builds and compares demand forecasting models for hourly borrowing and returning volumes. The models are evaluated using MAE, and the selected prediction results are exported for the dispatch optimization model.

### 3. Dispatch Optimization

The script `question3.py` compares no-dispatch and optimized-dispatch scenarios. The optimization model considers station capacity, predicted demand, transport cost, full-station penalties, empty-station penalties, and user service indicators.

## Technologies Used

* Python
* pandas
* NumPy
* matplotlib
* PuLP
* HiGHS

## How to Run

Install the required packages:

```bash
pip install pandas numpy matplotlib pulp highspy
```

Run the scripts in the following order:

```bash
python 统计图.py
python 预测模型.py
python question3.py
```

## Key Outputs

The project generates:

* Station demand analysis charts
* Prediction comparison results
* MAE comparison tables
* Workday demand prediction results
* No-dispatch vs optimized-dispatch comparison results
* Optimized bike dispatch plan

## What I Learned

Through this project, I gained practical experience in data analysis, demand forecasting, optimization modeling, and team-based mathematical modeling. I also learned how to connect prediction results with downstream operational decision-making.

## AI Assistance Statement

As this was my first mathematical modeling competition, I used AI tools as coding assistance for understanding implementation structures, debugging, and improving code organization. The final code logic, outputs, and interpretation were checked and discussed within the team.
