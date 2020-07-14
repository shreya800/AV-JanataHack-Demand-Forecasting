# AV-JanataHack-Demand-Forecasting
Approach to "Analytics Vidhya JanataHack Demand Forecasting"

Hosted at: https://datahack.analyticsvidhya.com/contest/janatahack-demand-forecasting/#ProblemStatement

Rank: 19 from 13,328 registered participant.

Problem Statement:
One of the largest retail chains in the world wants to use their vast data source to build an efficient forecasting model to predict the sales for each SKU in its portfolio at its 76 different stores using historical sales data for the past 3 years on a week-on-week basis. Sales and promotional information is also available for each week - product and store wise.

However, no other information regarding stores and products are available. Can you still forecast accurately the sales values for every such product/SKU-store combination for the next 12 weeks accurately? If yes, then dive right in!

Data Description:
Variable Definition
record_ID: Unique ID for each week store sku combination
week: Starting Date of the week
store_id: Unique ID for each store (no numerical order to be assumed)
sku_id: Unique ID for each product (no numerical order to be assumed)
total_price: Sales Price of the product 
base_price: Base price of the product
is_featured_sku: Was part of the featured item of the week
is_display_sku: Product was on display at a prominent place at the store
units_sold(Target): Total Units sold for that week-store-sku combination

Approach:
Generated following new features:
(a) Count of records per 'sku-id'
(b) Week number of the year (52 weeks)
(c) Week number of the month (4weeks)
(d) Quarter of the year(Q1, Q2, Q3, Q4)
(e) Day number of the year(365 days)
(f) Discount Percentage by (BasePrice-TotalPrice)/Base Price
(g) Discount Flag
(h) Count of records per 'store-id'
Did One hot encoding for 'sku-id' & 'store-id'
Then tuned xgboost Regressor Model
Trained the data on Xgboost
Tuned the above models
