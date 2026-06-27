<div align="center">

# < Divya /> 

### Building at the intersection of Data & Intelligence

Every line of code is a step forward.

---






</div>

---

##  What I Do Here

I don't just read tutorials.
I write code. I break things. I fix them. I push.

This repository is proof of that.

---

##  Day 1 — Pandas Series & DataFrame

> Explored how raw data becomes structured insight.

Built a script that takes student data and extracts
meaning from it — filtering, grouping, ranking, deriving.

*The moment that clicked:*
```python
df.groupby("City")["CGPA"].mean()
# One line. Entire story of a dataset.
**skills used:**
'pd.Series'.'pd.DataFrame'.'groupby'.'apply'.'lambda'.'describe'


### Day 2 — Student Result Analyser

> Learned how messy real-world data gets cleaned into insights.

Built a script that takes raw student data full of missing values
and duplicates — and cleans, ranks, and analyses it.

**The moment that clicked:**
```python
df["Rank"] = df["Marks"].rank(ascending=False).astype(int)
# Ranking 100 students in one line.
**Skills used:**
isnull · notnull · dropna · drop_duplicates · value_counts · sort_values · rank · set_index · reset_index 

### Day 3 - groupby(in progress)
> * Started understanding how groupby splits data and applies functions. *
** Learning today: **
'groupby()' . aggregating data by category .'.mean()' '.sum()' '.count()' on groups
** Status:** Still practicing - will add code once i build something with it.

### day 4- multi-source sales data consolidator (groupby,merging,joining,concatinate)
#  Multi-Source Sales Data Consolidator

A realistic data engineering problem — combining sales data scattered across 
multiple sources (quarterly sales, product catalog, customer records) into 
one unified business report.

## Problem This Solves
Real companies store data in separate systems — Sales team, Inventory team, 
and CRM each have their own tables. This project simulates combining them 
into actionable business insights.

##  Pandas Techniques Used
- `pd.concat()` — Combining quarterly data into one table
- `pd.merge()` — Joining sales with product and customer data (left joins)
- `.groupby().agg()` — Multi-metric analysis (sum, mean, count together)
- Business KPI calculation — Revenue, top customers, top products

## Insights Generated
- Total revenue and average order value
- Revenue breakdown by city
- Best-selling products
- Quarter-over-quarter performance
- Top spending customers

