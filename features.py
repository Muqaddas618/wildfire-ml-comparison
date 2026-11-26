"""
features.py - feature engineering and preprocessing shared by every model.
Kept in its own module so saved models can be loaded by predict_demo.py.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUM_COLS = ["LATITUDE", "LONGITUDE", "doy_sin", "doy_cos", "month", "weekday", "hour"]
CAT_COLS = ["STATE", "OWNER_DESCR", "NWCG_REPORTING_AGENCY", "SOURCE_SYSTEM_TYPE", "season"]
FREQ_COLS = ["FIPS_CODE"]            # high-cardinality county code -> frequency encoding

SIZE_LABELS = list("ABCDEFG")
SIZE_BINS = [-np.inf, 0.25, 9.95, 99.95, 299.95, 999.95, 4999.95, np.inf]  # NWCG size classes A-G


def size_class(fire_size):
    """NWCG size class A-G (used only for stratified sampling and error analysis)."""
    return pd.cut(fire_size, bins=SIZE_BINS, labels=SIZE_LABELS).astype(str)


def _season(month):
    return month.map({12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
                      6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"})


def build_features(df):
    """Raw FPA FOD columns -> model inputs. Needs FIRE_YEAR, DISCOVERY_DOY, DISCOVERY_TIME,
    LATITUDE, LONGITUDE, STATE, FIPS_CODE, OWNER_DESCR, NWCG_REPORTING_AGENCY, SOURCE_SYSTEM_TYPE."""
    out = pd.DataFrame(index=df.index)
    doy = pd.to_numeric(df["DISCOVERY_DOY"], errors="coerce")
    out["LATITUDE"] = pd.to_numeric(df["LATITUDE"], errors="coerce")
    out["LONGITUDE"] = pd.to_numeric(df["LONGITUDE"], errors="coerce")
    # day of year as sine/cosine so that December and January are close
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    # calendar date rebuilt from FIRE_YEAR + day of year (same information as DISCOVERY_DATE)
    date = pd.to_datetime(df["FIRE_YEAR"].astype(int).astype(str) + "-01-01") + pd.to_timedelta(doy - 1, unit="D")
    out["month"] = date.dt.month
    out["weekday"] = date.dt.weekday
    out["season"] = _season(out["month"])
    # DISCOVERY_TIME is hhmm; missing or invalid -> NaN (imputed later, with a missing flag)
    if "DISCOVERY_TIME" in df:
        hour = np.floor(pd.to_numeric(df["DISCOVERY_TIME"], errors="coerce") / 100)
        hour = hour.where((hour >= 0) & (hour <= 23))
    else:
        hour = pd.Series(np.nan, index=df.index)
    out["hour"] = hour
    for c in ["STATE", "OWNER_DESCR", "NWCG_REPORTING_AGENCY", "SOURCE_SYSTEM_TYPE", "FIPS_CODE"]:
        out[c] = df[c].astype(str).replace({"None": "Unknown", "nan": "Unknown", "": "Unknown"}) if c in df else "Unknown"
    return out


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replaces a category by log(1 + number of training rows with that category)."""

    def fit(self, X, y=None):
        self.counts_ = pd.Series(np.asarray(X).ravel().astype(str)).value_counts()
        return self

    def transform(self, X):
        s = pd.Series(np.asarray(X).ravel().astype(str))
        return np.log1p(s.map(self.counts_).fillna(0).to_numpy(dtype=float)).reshape(-1, 1)

    def get_feature_names_out(self, names=None):
        return np.array(["county_freq"])


def make_preprocessor(scale, num_cols=NUM_COLS, cat_cols=CAT_COLS, freq_cols=FREQ_COLS):
    """Same preprocessing for all models. Scaling is switched on only for the linear models."""
    num_steps = [("imp", SimpleImputer(strategy="median", add_indicator=True))]
    freq_steps = [("freq", FrequencyEncoder())]
    if scale:
        num_steps.append(("sc", StandardScaler()))
        freq_steps.append(("sc", StandardScaler()))
    parts = [
        ("num", Pipeline(num_steps), list(num_cols)),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.005,
                              sparse_output=False), list(cat_cols)),
    ]
    if freq_cols:
        parts.append(("freq", Pipeline(freq_steps), list(freq_cols)))
    return ColumnTransformer(parts)
