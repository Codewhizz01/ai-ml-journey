import pandas as pd

# ── SOURCE 1: Sales Transactions (from Sales team) ──
sales_q1 = pd.DataFrame({
    "OrderID": [101, 102, 103, 104],
    "CustomerID": [1, 2, 1, 3],
    "ProductID": ["P1", "P2", "P3", "P1"],
    "Quantity": [2, 1, 3, 1],
    "Quarter": ["Q1", "Q1", "Q1", "Q1"]
})

sales_q2 = pd.DataFrame({
    "OrderID": [105, 106, 107],
    "CustomerID": [2, 3, 4],
    "ProductID": ["P2", "P3", "P1"],
    "Quantity": [2, 1, 4],
    "Quarter": ["Q2", "Q2", "Q2"]
})

#  Product Catalog (from Inventory team)
products = pd.DataFrame({
    "ProductID": ["P1", "P2", "P3"],
    "ProductName": ["Laptop", "Mouse", "Keyboard"],
    "Price": [55000, 800, 1500]
})

#  Customer Info (from CRM team)
customers = pd.DataFrame({
    "CustomerID": [1, 2, 3, 4],
    "CustomerName": ["Riya", "Priya", "Sneha", "Anjali"],
    "City": ["Delhi", "Mumbai", "Pune", "Delhi"]
})

print("=" * 60)
print("STEP 1 — CONCATENATE: Combine Q1 + Q2 sales into one table")
print("=" * 60)
all_sales = pd.concat([sales_q1, sales_q2], ignore_index=True)
print(all_sales)

print("\n" + "=" * 60)
print("STEP 2 — MERGE: Add Product details to sales")
print("=" * 60)
sales_with_products = pd.merge(all_sales, products, on="ProductID", how="left")
print(sales_with_products)

print("\n" + "=" * 60)
print("STEP 3 — MERGE: Add Customer details to sales")
print("=" * 60)
full_data = pd.merge(sales_with_products, customers, on="CustomerID", how="left")
print(full_data)

print("\n" + "=" * 60)
print("STEP 4 — Calculate Total Revenue per Order")
print("=" * 60)
full_data["TotalRevenue"] = full_data["Quantity"] * full_data["Price"]
print(full_data[["OrderID", "CustomerName", "ProductName", "Quantity", "TotalRevenue"]])

print("\n" + "=" * 60)
print("STEP 5 — JOIN-style Analysis: Revenue by City")
print("=" * 60)
city_revenue = full_data.groupby("City")["TotalRevenue"].sum().sort_values(ascending=False)
print(city_revenue)

print("\n" + "=" * 60)
print("STEP 6 — Revenue by Product")
print("=" * 60)
product_revenue = full_data.groupby("ProductName")["TotalRevenue"].sum().sort_values(ascending=False)
print(product_revenue)

print("\n" + "=" * 60)
print("STEP 7 — Quarter-wise Performance")
print("=" * 60)
quarter_performance = full_data.groupby("Quarter")["TotalRevenue"].agg(["sum", "mean", "count"])
quarter_performance.columns = ["Total Revenue", "Avg Order Value", "Order Count"]
print(quarter_performance)

print("\n" + "=" * 60)
print("STEP 8 — Top Customer by Spending")
print("=" * 60)
customer_spending = full_data.groupby("CustomerName")["TotalRevenue"].sum().sort_values(ascending=False)
print(customer_spending)
top_customer = customer_spending.idxmax()
print(f"\n Top Customer: {top_customer} (₹{customer_spending.max()})")

print("\n" + "=" * 60)
print("FINAL BUSINESS SUMMARY")
print("=" * 60)
print(f"Total Orders: {len(full_data)}")
print(f"Total Revenue: ₹{full_data['TotalRevenue'].sum()}")
print(f"Average Order Value: ₹{full_data['TotalRevenue'].mean():.2f}")
print(f"Best Performing City: {city_revenue.idxmax()}")
print(f"Best Selling Product: {product_revenue.idxmax()}")