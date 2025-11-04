# Phase02 Report



### 1. Cancellation Rates for each month

We calculated the percentage of bookings canceled in each month by grouping on `arrival_month` and computing the mean of the binary `is_canceled` column. Higher percentages indicate months with heavier booking volatility.


```python

    cancel_rate_by_month = (
        #group data by arrival month
        df.groupby('arrival_month')['is_canceled']
        # sum up binary column and divide by number of samples
        .mean()
        # convert to percentage
        .mul(100)
        .reset_index(name='cancel_rate (%)')
        .sort_values('arrival_month')
    )

```


| arrival_month | cancel_rate (%) |
|----------------|-----------------|
| 1              | 17.811159       |
| 2              | 31.581769       |
| 3              | 30.311891       |
| 4              | 37.432631       |
| 5              | 35.450718       |
| 6              | 39.870512       |
| 7              | 39.657187       |
| 8              | 38.472385       |
| 9              | 37.396653       |
| 10             | 37.179098       |
| 11             | 30.662983       |
| 12             | 28.293031       |

![Cancel Rate by Month](Cancel_Rate_by_month.png)

Our data shows that rates spikes in summer months (June-August) and are lowest in January and December. 

### 2. Compute Average price and average number of nights for each month.


For each month, we computed:
- `avg_price_per_room` — mean daily booking revenue
- `nights_stayed` — total weekday + weekend nights

| arrival_month | avg_price_per_room | stayed_nights |
| ------------- | ------------------ | ------------- |
| 1             | 67.870616          | 2.738197      |
| 2             | 73.340136          | 2.877033      |
| 3             | 83.276551          | 3.210805      |
| 4             | 93.302379          | 3.117834      |
| 5             | 102.000425         | 3.133358      |
| 6             | 108.858464         | 3.157151      |
| 7             | 114.074734         | 3.758181      |
| 8             | 122.761549         | 3.645985      |
| 9             | 108.360909         | 3.288313      |
| 10            | 93.250079          | 3.043879      |
| 11            | 79.127639          | 3.134131      |
| 12            | 83.504243          | 3.252321      |


![scaled avg price and nights stayed comparison](scaled_comparison.png)

Prices and stay lengths both peak in July–August, which tracks with vacation season. When rooms cost more, guests stay longer 

### 3. Count monthly booking by market segment. 

We counted bookings by month and `market_segment_type` (Direct, Corporate, Groups, Online TA/TO, etc.).

- TA: Travel Agents
- TO: Tour Operators

![Monthly Bookings by Market Segment](MMS.png)

Online channels dominate every month, especially toward the end of the year.

### 4. Identify the most popular month of the year for bookings based on revenue

Most popular month is September $1638308.59

![Monthly Revenune by Bookings](Monthly_Revenue_bookings.png)

