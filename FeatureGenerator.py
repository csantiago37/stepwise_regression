import itertools
import pandas as pd
import numpy as np
import holidays
from pandas.tseries.offsets import MonthBegin
from sklearn.preprocessing import PolynomialFeatures


class FeatureGenerator:

    def __init__(self, x, x_dt):
        self.x = x
        self.x_dt = x_dt
        self.interactions = None
        self.lag_features = None
        self.window_features = None
        self.change_features = None
        self.date_features = None
        self.days_features = None
        self.new_x = None

    def create_interactions(self):
        interactions = PolynomialFeatures(interaction_only=True)
        new_features = interactions.fit_transform(self.x.fillna(0))
        old_cols = list(self.x.columns)
        new_cols = [':'.join(x) for x in itertools.combinations(self.x.columns, r=2)]
        
        # first col from polynomials is constant; dropping
        df = pd.DataFrame(new_features).iloc[:, 1:]
        df.columns = old_cols + new_cols
        
        self.interactions = df.to_dict(orient='list')
        return df      

    def create_lag_features(self, n_lags):
        new_features = {}
        for col in self.x.columns:
            for lag in range(1, n_lags+1):
                col_name = col + f'_lag{lag}'
                col_value = self.x[col].shift(lag)
                new_features[col_name] = col_value

        self.lag_features = new_features
        df = pd.DataFrame(new_features)
        return df

    def create_window_features(self, window_lengths):
        funcs = {
        'sum': np.sum,
        'mean': np.mean,
        'std': np.std,
        'min': np.min,
        'max': np.max,
        }

        new_features = {}
        for col in self.x.columns:
            for window in window_lengths:
                for func_name, func_call in funcs.items():
                    col_name = col + f'_rolling_{window}_period_{func_name}'
                    col_value = self.x[col].rolling(window).apply(func_call, raw=False)
                    new_features[col_name] = col_value

        self.window_features = new_features
        df = pd.DataFrame(new_features)
        return df

    def create_change_features(self, period_lengths):
        new_features = {}
        for col in self.x.columns:
            for period in period_lengths:

                col_name = col + f'_{period}_period_pct_chg'
                col_value = self.x[col].pct_change(periods=period)
                new_features[col_name] = col_value

                col_name = col + f'_{period}_period_diff'
                col_value = self.x[col].diff(periods=period)
                new_features[col_name] = col_value

        self.change_features = new_features
        df = pd.DataFrame(new_features)
        return df

    def create_date_features(self, dt_format=None):
        new_features = {}

        if dt_format is not None:
            x_dt = pd.to_datetime(self.x_dt, format=dt_format)
        else:
            x_dt = pd.to_datetime(self.x_dt)

        new_features['day'] = x_dt.dt.day
        new_features['week'] = x_dt.dt.week
        new_features['month'] = x_dt.dt.month
        new_features['quarter'] = x_dt.dt.quarter
        new_features['year'] = x_dt.dt.year

        self.date_features = new_features
        df = pd.DataFrame(new_features)
        return df

    def create_days_features(self, dt_format=None):
        new_features = {}

        if dt_format is not None:
            x_dt = pd.to_datetime(self.x_dt, format=dt_format)
        else:
            x_dt = pd.to_datetime(self.x_dt)

        holiday_years = list(range(2000, 2050))
        us_holidays = [x for x in holidays.US(years=holiday_years)]

        def get_bus_days(dt):
            eom = dt
            bom = eom - MonthBegin()
            bus_days = np.busday_count(bom.date(), eom.date(), holidays=us_holidays)
            return bus_days + 1  # np.busday_count is exclusive of ending date

        new_features['days'] = x_dt.dt.daysinmonth
        new_features['bus_days'] = x_dt.apply(get_bus_days)

        self.days_features = new_features
        df = pd.DataFrame(new_features)
        return df

    def create_all_x(self, n_lags, window_lengths, period_lengths, add_dt=False):
        x_lags = self.create_lag_features(n_lags)
        x_windows = self.create_window_features(window_lengths)
        x_changes = self.create_change_features(period_lengths)

        if add_dt:
            x_date = self.create_date_features()
            x_days = self.create_days_features()
            df = pd.concat([self.x, x_lags, x_windows, x_changes, x_date, x_days], axis=1)
        else:
            df = pd.concat([self.x, x_lags, x_windows, x_changes], axis=1)

        self.new_x = df
        return df

    def merge_xy(self, y, drop_na=True):
        df = pd.concat([y, self.new_x], axis=1)
        if drop_na:
            df.dropna(inplace=True)
            df.reset_index(inplace=True, drop=True)

        x = df.iloc[:, 1:]
        y = df.iloc[:, 0]
        return x, y
