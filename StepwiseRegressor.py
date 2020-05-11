from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import pandas as pd
from sklearn.preprocessing import StandardScaler


class StepwiseRegressor:
    def __init__(self, df, response, model=None):
        self.response = response
        self._data = df
        self._index = df.index
        self.x = self._get_x
        self.y = self._get_y

        if model is None:
            self.model = OLS(endog=self.y, exog=self.x)
        else:
            self.model = model

        self.null_model = self.get_null_model()
        self.results = {}

    @property
    def _get_x(self):
        x_cols = [col for col in self._data.columns if self.response not in col]
        raw_x = self._data[x_cols]
        sc_x = self._standardize(raw_x)
        full_x = add_constant(sc_x)
        full_x.index = self._index
        return full_x

    @staticmethod
    def _standardize(x):
        feature_names = x.columns
        sclr = StandardScaler()
        sc_x = sclr.fit_transform(x)
        sc_x = pd.DataFrame(sc_x, columns=feature_names)
        return sc_x

    @property
    def _get_y(self):
        y_col = [self.response]
        y = self._data[y_col]
        y.index = self._index
        return y

    def _trial_model(self, features):
        y = self.y
        x = self.x[features]
        model = OLS(endog=y, exog=x)
        fit = model.fit()
        return fit

    def get_null_model(self):
        const = self.x[['const']]
        model = OLS(endog=self.y, exog=const)
        null_results = model.fit()
        return null_results

    def forward_selection(self, use_aic=True):
        best_model = self.null_model
        if use_aic:
            best_metric = best_model.aic
        else:
            best_metric = best_model.bic
        test_metric = 0
        current_cols = ['const']
        while test_metric <= best_metric:
            all_features = [
                col for col in self.x.columns if col not in current_cols]
            if len(all_features) == 0:
                break
            scores = []
            trials = []
            for col in all_features:
                trial_cols = current_cols + [col]
                trial_fit = self._trial_model(trial_cols)
                if use_aic:
                    trial_metric = trial_fit.aic
                else:
                    trial_metric = trial_fit.bic
                scores.append(trial_metric)
                trials.append(trial_fit)

            min_score = min(scores)
            min_idx = scores.index(min_score)

            test_metric = min_score
            if test_metric < best_metric:
                best_metric = test_metric
                best_model = trials[min_idx]
                add_col = all_features[min_idx]
                current_cols.append(add_col)
            else:
                break

        return best_model

    def fit(self):
        pass
