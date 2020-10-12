import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
import joblib


data = pd.read_csv("data/VCPe SevOne Data 10_7 10_8 7am with Deltas and disk.csv")
data = data[['DEVICENAME', 'SSCPUIDLE', 'TIME']]
data['TIME'] = pd.to_datetime(pd.to_datetime(data['TIME'], unit='s').dt.strftime("%Y-%m-%d %H:00"))
data['Day'] = data.TIME.dt.date
data['Hour'] = data.TIME.dt.hour
# data.index = data.TIME
data = data.drop(columns=['TIME'])

imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
d = {}
by_device = data.groupby('DEVICENAME')
for device, device_df in by_device:
    print(device)
    by_day = device_df.groupby('Day')
    for day, day_df in by_day:
        if 24 > len(day_df) > 12:
            day_df = day_df.sort_values('SSCPUIDLE').drop_duplicates(subset=['Day', 'Hour'], keep='last').sort_values('Hour')
            day_df.index = day_df.Hour
            new_df = pd.DataFrame(index=list(range(0, 24)), columns=day_df.columns)
            new_df.update(day_df)
            new_df.DEVICENAME = device
            new_df.Day = day
            new_df.Hour = new_df.index
            new_df.SSCPUIDLE = imp_mean.fit_transform(new_df.SSCPUIDLE.values.reshape(-1, 1))
            d.update({
                device: [device, day, new_df.SSCPUIDLE.values]
            })
        else:
            continue

df = pd.DataFrame.from_dict(d, orient='index', columns=['Device', 'Day', 'TimeSeries']).reset_index(drop=True)
df['Location'] = df.Device.str.extract(r'([^-]*).*')
ts_data = df.reindex(['Location', 'Device', 'Day', 'TimeSeries'], axis=1)


joblib.dump([data, ts_data], "data/data.job")