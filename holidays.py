import holidays
import pandas as pd
from collections import Counter
data=[]
for year in range(2015,2024):
    portugal_holidays = holidays.Portugal(year)
    holiday_months = [date.month for date in portugal_holidays]
    holiday_count = Counter(holiday_months)
    for month in range(1, 13):
        text=f"{year},{month},{holiday_count.get(month, 0)}"
        print(text)
        data.append(text)
table= pd.DataFrame(data)
table.to_csv('./Feriados.csv', index=False)
input('Press ENTER to exit')