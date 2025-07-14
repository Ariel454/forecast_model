import pandas as pd
import pathlib as pl

data_dir = pl.Path("data")

list(data_dir.glob("*.csv"))
def load_and_melt_variable(file_path: pl.Path, variable_name: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df = df.rename(columns={"Date_time": "datetime"})
    df = df.melt(id_vars=["datetime"], var_name="station", value_name="value")
    df["variable"] = variable_name
    return df


# Diccionario: variable -> nombre de archivo
variable_files = {
    "CO": "CO.csv",
    "DIR": "DIR.csv",
    "HUM": "HUM.csv",
    "LLU": "LLU.csv",
    "NO2": "NO2.csv",
    "O3": "O3.csv",
    "PM2.5": "PM2.5.csv",
    "PM10": "PM10.csv",
    "PRE": "PRE.csv",
    "RS": "RS.csv",
    "SO2": "SO2.csv",
    "TMP": "TMP.csv",
    "VEL": "VEL.csv"
}

# Lista para almacenar todos los DataFrames en formato largo
long_dfs = []

for var, filename in variable_files.items():
    path = data_dir / filename
    df = load_and_melt_variable(path, var)
    long_dfs.append(df)

# Concatenar todos en uno solo
consolidated_df = pd.concat(long_dfs, ignore_index=True)
consolidated_df["datetime"] = pd.to_datetime(consolidated_df["datetime"], errors="coerce")
consolidated_df.dropna(subset=["datetime"], inplace=True)
consolidated_df.head()
# Pivotear a DataFrame ancho con columnas combinadas: variable + estación
wide_df = consolidated_df.pivot_table(
    index="datetime",
    columns=["variable", "station"],
    values="value"
)

# Opcional: renombrar columnas tipo "PM2.5_BELISARIO"
wide_df.columns = [f"{var}_{st}" for var, st in wide_df.columns]
wide_df = wide_df.sort_index()

wide_df.head()
import matplotlib.pyplot as plt
import missingno as msno


plt.figure(figsize=(16, 6))
msno.matrix(wide_df, sparkline=False, fontsize=8)
plt.title("Valores faltantes en todo el dataset (todas las variables y estaciones)")
plt.show()
import collections as ct

station_counter = ct.Counter()
stations_by_file = {}

for var, fname in variable_files.items():
    cols = pd.read_csv(data_dir / fname, nrows=0).columns
    stations = [c.replace('_', ' ').upper().strip() for c in cols if c != "Date_time"]
    station_counter.update(stations)
    stations_by_file[var] = stations

# ---------- 1) resumen global ----------
print("Frecuencia de cada estación en los 13 CSV:")
for st, n in station_counter.most_common():
    print(f"{st:12s}  aparece en {n:2d} archivos")

# ---------- 2) ¿está CARAPUNGO en todos? ----------
total_files = len(variable_files)
print(f"\nCARAPUNGO figura en {station_counter['CARAPUNGO']} de {total_files} archivos")

# ---------- 3) tabla archivo → estaciones (para informe) ----------
file_df = (
    pd.Series(stations_by_file)
      .explode()
      .reset_index()
      .rename(columns={'index':'variable', 0:'station'})
      .pivot_table(index='station', columns='variable', aggfunc='size', fill_value='')
)

target_station = "CARAPUNGO"          # estación de interés
zeros_to_nan   = ["PM2.5", "PM10", "CO", "NO2", "SO2", "O3"]  # 0 = “sin dato”

# ─── 2. Construir un dict {variable: Series} sólo con CARAPUNGO ──────────────
series_dict = {}
for var, fname in variable_files.items():
    df = pd.read_csv(data_dir / fname, usecols=["Date_time", target_station])
    df = (df.rename(columns={"Date_time": "datetime", target_station: var})
            .assign(datetime=lambda d: pd.to_datetime(d["datetime"], errors="coerce"))
            .dropna(subset=["datetime"])
            .set_index("datetime")
            .sort_index())
    if var in zeros_to_nan:                         # 0  →  NaN en ciertas variables
        df.loc[df[var] == 0, var] = pd.NA
    series_dict[var] = df[var]

carapungo_df = pd.concat(series_dict, axis=1)      # (3) combinar variables

# ─── 2. Reindexar a malla horaria completa ──────────────────────────────────
full_range   = pd.date_range(carapungo_df.index.min(),
                             carapungo_df.index.max(), freq="h")
carapungo_df = carapungo_df.reindex(full_range)

# ─── 3. Visualizar con ticks anuales en el eje Y ────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
msno.matrix(carapungo_df, sparkline=False, fontsize=8, ax=ax)
ax.set_title("Valores faltantes por variable – Estación CARAPUNGO (malla horaria)")

# Generamos un tick por 1.º de enero de cada año
years  = pd.date_range(carapungo_df.index.min().normalize(),
                       carapungo_df.index.max().normalize(),
                       freq="YS")                      # Year Start
rowpos = ((years - carapungo_df.index.min()) / pd.Timedelta("1h")).astype(int)
ax.set_yticks(rowpos)
ax.set_yticklabels(years.year)
ax.set_ylabel("Año")
plt.show()

# ─── 4. Resumen numérico de cobertura ───────────────────────────────────────
coverage = (100 - carapungo_df.isna().mean()*100).round(2).sort_values(ascending=False)
print("Cobertura (%) por variable – CARAPUNGO\n", coverage)

import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno

# ------------------------------------------------------------
# 2. Seleccionar variables con ≥60 % de cobertura
#    y descartar explícitamente NO2 y PM10
# ------------------------------------------------------------
threshold   = 0.60                               # 60 %
good_cols   = carapungo_df.columns[
                 carapungo_df.notna().mean() >= threshold
             ].difference(["NO2", "PM10"])       # quita NO2 y PM10
df_good     = carapungo_df[good_cols]

# ------------------------------------------------------------
# 3. Recortar ventana 2008-01-01 → 2017-12-31
# ------------------------------------------------------------
df_good = df_good.loc["2008":"2017"]

# ------------------------------------------------------------
# 4. Visualizar huecos (missingno) con ticks anuales
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 6))
msno.matrix(df_good, fontsize=8, sparkline=False, ax=ax)

ax.set_title("Huecos tras filtrar (CARAPUNGO · 2008-2017, sin NO₂/PM10)")

# ticks: 1.º-ene de cada año
years   = pd.date_range("2008-01-01", "2017-01-01", freq="YS")
rowpos  = ((years - df_good.index.min()) / pd.Timedelta("1h")).astype(int)
ax.set_yticks(rowpos)
ax.set_yticklabels(years.year)
ax.set_ylabel("Año")
plt.show()

# ------------------------------------------------------------
# 5. Cobertura final para el informe
# ------------------------------------------------------------
coverage = (100 - df_good.isna().mean()*100).round(2).sort_values(ascending=False)
print("Cobertura (%) después del filtrado:\n", coverage)


import numpy as np, pandas as pd, torch
from pypots.imputation.saits import SAITS

# -------------------------------------------------------
# 0.  df_good : tu DataFrame (index = horas 2008-2017)
# -------------------------------------------------------
X_raw  = df_good.to_numpy(np.float32)        # (T, F)
mask   = (~np.isnan(X_raw)).astype(np.float32)

# ---------------- normalización -----------------------
mu, sig = np.nanmean(X_raw, 0), np.nanstd(X_raw, 0)
X_norm  = (X_raw - mu) / sig                 # (T, F)

# ---------------- windowing ---------------------------
WINDOW  = 24*7*4        # 4 semanas  (672 pasos)
STRIDE  = 24*7          # 1 semana   (168 pasos)

Xs, Ms = [], []
for s in range(0, len(X_norm) - WINDOW + 1, STRIDE):
    Xs.append( X_norm[s:s+WINDOW] )
    Ms.append( mask [s:s+WINDOW] )

X_batch   = np.stack(Xs)                     # (N, T, F)
mask_batch= np.stack(Ms)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("PyTorch ve CUDA:", torch.cuda.is_available(), "| usando:", device)


# ---------------- modelo SAITS ------------------------
T, F = X_batch.shape[1:]
model = SAITS(
    n_steps    = T,
    n_features = F,
    d_model    = 64,
    d_ffn      = 128,
    n_layers   = 4,
    n_heads    = 4,
    d_k        = 16, d_v = 16,
    dropout    = 0.1,
    batch_size = 32,
    epochs     = 200,
    patience   = 10,
    device     = "cuda" if torch.cuda.is_available() else "cpu",
)

# ---------------- entrenamiento ------------------------

train_dict = {"X": X_batch, "missing_mask": mask_batch}
model.fit(train_dict)

print("Memoria GPU tras entrenamiento (MB):",
      torch.cuda.memory_allocated() / 1024**2)

# ---------------- imputación --------------------------
# ↓↓↓ Mueve a CPU ↓↓↓
model.device = "cpu"
model.model.to("cpu")

# ahora sí imputas sin reentrenar
out = model.impute(train_dict)  # ya es ndarray (N, T, F)
X_imp_norm  = np.zeros_like(X_norm)
count_hits  = np.zeros_like(X_norm)

for k, s in enumerate(range(0, len(X_norm) - WINDOW + 1, STRIDE)):
    X_imp_norm[s:s+WINDOW] += out[k]
    count_hits[s:s+WINDOW] += 1

X_imp_norm /= count_hits
X_imp       = X_imp_norm * sig + mu

imputed_df = pd.DataFrame(X_imp, index=df_good.index, columns=df_good.columns)
imputed_df.to_parquet("data/processed/carapungo_imputed_hourly.parquet")
print("✓ Imputación finalizada y guardada")


import pandas as pd

df = pd.read_parquet("data/processed/carapungo_imputed_hourly.parquet")
print(df.head())              # primeras 5 filas
print(df.info())              # resumen de columnas y tipos

msno.matrix(imputed_df, figsize=(14, 6), fontsize=8, sparkline=False)
plt.title("SAITS · Imputación de CARAPUNGO (2008-2017)")
plt.show()

import pandas as pd
from pathlib import Path

# ------------------------------------------------------------------
# 1. Cargar la serie imputada horario 2008-2017  (CARAPUNGO)
# ------------------------------------------------------------------
data_dir       = Path("data/processed")
hourly_file    = data_dir / "carapungo_imputed_hourly.parquet"
df_hourly      = pd.read_parquet(hourly_file)

# aseguramos que el índice es DatetimeIndex y está ordenado
df_hourly.index = pd.to_datetime(df_hourly.index)
df_hourly = df_hourly.sort_index()

# ------------------------------------------------------------------
# 2. Mapeo de funciones de agregación por variable
#    - LLU (precipitación) se **suma**
#    - el resto se promedia
# ------------------------------------------------------------------
agg_map = {col: ("sum" if col == "LLU" else "mean") for col in df_hourly.columns}

# ------------------------------------------------------------------
# 3. Resampleo diario
# ------------------------------------------------------------------
df_daily = df_hourly.resample("D").agg(agg_map)

# 4. Persistir y revisar (raw)
# ------------------------------------------------------------------
df_daily.to_parquet("data/processed/carapungo_imputed_daily_raw.parquet", compression="snappy")
print("Shape diario raw:", df_daily.shape)
print(df_daily.head())

# ─── NUEVO BLOQUE: detectar y eliminar días sin PM2.5 ─────────────
missing_days = df_daily[df_daily["PM2.5"].isna()].index
print("Fechas sin PM2.5, se borran:", missing_days.tolist())

df_daily_clean = df_daily.drop(index=missing_days)

# ------------------------------------------------------------------
# 5. Guardar Parquet diario limpio
# ------------------------------------------------------------------
df_daily_clean.to_parquet("data/processed/carapungo_imputed_daily.parquet", compression="snappy")
print("Shape diario limpio:", df_daily_clean.shape)


import pandas as pd
from pathlib import Path



### ─── EDA ────────────────────────────────

# 1.1. Leer el parquet diario de Carapungo
daily_file = Path("data/processed") / "carapungo_imputed_daily.parquet"
df = pd.read_parquet(daily_file)

# 1.2. Asegurar índice datetime
df.index = pd.to_datetime(df.index)
df = df.sort_index()

print("Fechas:", df.index.min(), "→", df.index.max())
print("Variables:", list(df.columns))


import pandas as pd
from pathlib import Path

# ─── 6. Leer el parquet diario limpio y descomponer ───────────────
daily_file = Path("data/processed") / "carapungo_imputed_daily.parquet"
df = pd.read_parquet(daily_file)
df.index = pd.to_datetime(df.index)
df = df.sort_index()

print("Fechas:", df.index.min(), "→", df.index.max())
print("Variables:", list(df.columns))

import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# 2.1. Time series lineplot de PM2.5
plt.figure(figsize=(14,4))
plt.plot(df.index, df["PM2.5"], label="PM2.5")
plt.title("Serie diaria de PM2.5 (Carapungo)")
plt.ylabel("µg/m³")
plt.xlabel("Fecha")
plt.legend()
plt.show()

# 2.2. Descomposición aditiva
decomp = seasonal_decompose(df["PM2.5"], model="additive", period=365)
fig = decomp.plot()
fig.set_size_inches(14,8)
plt.show()

import seaborn as sns

# 3.1. Matriz de correlación
corr = df.corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
plt.title("Correlaciones diarias – Carapungo (2008–2017)")
plt.show()


from prophet import Prophet

# Partiendo de tu prophet_df actual:
prophet_df = prophet_df.rename(columns={"index": "ds"})

# Si queda alguna fila con y nulo, descartarla:
prophet_df = prophet_df.dropna(subset=["y"])

# Ahora sí tienes las columnas correctas:
print(prophet_df.columns)  # ['ds', 'y']

# Entrenas Prophet:
m = Prophet(yearly_seasonality=True, changepoint_prior_scale=0.5)
m.fit(prophet_df)

# Haces el forecast in‐sample
future = m.make_future_dataframe(periods=0, freq="D")
forecast = m.predict(future)

# Y finalmente lo ploteas:
fig = m.plot(forecast, figsize=(12,6))
plt.title("Detección de changepoints en PM2.5 (Prophet)")
plt.show()


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 5.1. Normalizar antes de PCA/t-SNE
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(df)

# 5.2. PCA (2 componentes)
pca = PCA(2)
xp = pca.fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(xp[:,0], xp[:,1], c=df.index.year, cmap="tab10", s=15)
plt.colorbar(label="Año")
plt.title("PCA 2D de variables diarias (coloreado por año)")
plt.show()

# 5.3. t-SNE (opcional, puede tardar)
tsne = TSNE(2, random_state=0)
xt = tsne.fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(xt[:,0], xt[:,1], c=df.index.year, cmap="tab10", s=15)
plt.colorbar(label="Año")
plt.title("t-SNE de variables diarias (coloreado por año)")
plt.show()

# df = carapungo_imputed_daily  (2008–2017, imputado con SAITS)
keep = ["CO","DIR","HUM","PRE","RS","TMP","VEL","PM2.5"]
model_df = df[keep].copy()

# revisamos rápidamente
print("Variables finales:", model_df.columns.tolist())
model_df.info()

import numpy as np
import pandas as pd

# partimos de tu DataFrame diario imputado
fe_df = model_df.copy()

# ─── 1) Variables de calendario ────────────────────────────────────────────
fe_df["month"]      = fe_df.index.month
fe_df["dayofyear"]  = fe_df.index.dayofyear
fe_df["dayofweek"]  = fe_df.index.dayofweek
fe_df["weekofyear"] = fe_df.index.isocalendar().week

# ciclos suaves (sin/cos) sobre dayofyear para capturar la estacionalidad anual
fe_df["sin_doy"] = np.sin(2 * np.pi * fe_df["dayofyear"] / 365.0)
fe_df["cos_doy"] = np.cos(2 * np.pi * fe_df["dayofyear"] / 365.0)

# ─── 2) Lag features de PM2.5 ──────────────────────────────────────────────
for lag in (1, 2, 3, 7, 14):
    fe_df[f"PM2.5_lag{lag}"] = fe_df["PM2.5"].shift(lag)

# ─── 3) Rolling means ──────────────────────────────────────────────────────
# ventana de 7 y 30 días
fe_df["PM2.5_roll7"]   = fe_df["PM2.5"].rolling(7,  min_periods=1).mean()
fe_df["CO_roll7"]      = fe_df["CO"].rolling(7,   min_periods=1).mean()
fe_df["TMP_roll7"]     = fe_df["TMP"].rolling(7,  min_periods=1).mean()

fe_df["PM2.5_roll30"]  = fe_df["PM2.5"].rolling(30, min_periods=1).mean()
fe_df["PRE_roll30"]    = fe_df["PRE"].rolling(30,  min_periods=1).sum()   # lluvia acumulada mensual

# ─── 4) Dummies de grandes eventos o changepoints ─────────────────────────
# (quita o adapta estas fechas según lo veas en tu gráfico de Prophet)
changepoints = ["2009-10-01", "2013-08-01", "2017-07-01"]
for cp in changepoints:
    fe_df[f"after_{cp}"] = (fe_df.index >= pd.to_datetime(cp)).astype(int)

# ─── 5) Limpiar filas con NA introducidos por los lags ────────────────────
fe_df = fe_df.dropna()

# ─── 6) Listado final de columnas y comprobación ──────────────────────────
print("Número de observaciones:", len(fe_df))
print("Variables featureadas:", fe_df.columns.tolist())


## Model building and testing

import pandas as pd
import numpy as np

# --- 0) preparar datos ya featureados -----------------------------------
# fe_df: DataFrame con columnas ['CO','DIR','HUM','PRE','RS','TMP','VEL','PM2.5', ...lags/rolls...]
# índice diario desde 2008-01-01 a 2017-10-16 (N=3563)
fe_df = pd.read_parquet("data/processed/carapungo_featured_daily.parquet")

# separar target y features
y = fe_df["PM2.5"]
X = fe_df.drop(columns=["PM2.5"])


import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

# 1) Serie mensual
monthly = df["PM2.5"].resample("ME").mean()

# 2) Train/test split
train_m = monthly[: "2016-12"]
test_m  = monthly["2017-01":]
n_test  = len(test_m)

# 3) Ajustar SARIMAX
model = SARIMAX(
    train_m,
    order=(1,1,1),
    seasonal_order=(1,1,1,12),
    enforce_stationarity=False,
    enforce_invertibility=False
)
res = model.fit(disp=False)

# 4) Forecast
fcst_m = res.get_forecast(steps=n_test).predicted_mean
fcst_m.index = test_m.index

# 5) Métricas sin usar `squared=`
mse  = mean_squared_error(test_m, fcst_m)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(test_m, fcst_m)

print(f"SARIMAX Monthly → RMSE: {rmse:.2f}, MAE: {mae:.2f}")


SARIMAX Monthly → RMSE: 143.73, MAE: 78.20

import matplotlib.pyplot as plt

plt.figure(figsize=(10,4))
plt.plot(test_m.index, test_m, label="Observado", marker="o")
plt.plot(fcst_m.index, fcst_m, label="Predicho SARIMAX", marker="x")
plt.title("Pronóstico mensual de PM2.5 (Carapungo) – SARIMAX")
plt.ylabel("PM2.5 [µg/m³]")
plt.xlabel("Fecha")
plt.legend()
plt.tight_layout()
plt.show()


# Supón que tu modelo SARIMAX está ajustado a toda la serie
years = 5
n_steps = years * 12  # meses

future_pred = res.get_forecast(steps=n_steps).predicted_mean

# Fechas del forecast
future_dates = pd.date_range(monthly.index[-1] + pd.offsets.MonthEnd(1), periods=n_steps, freq="M")
future_pred.index = future_dates

# Graficar junto a histórico
plt.figure(figsize=(12,5))
plt.plot(monthly.index, monthly, label="Histórico")
plt.plot(future_pred.index, future_pred, label="Forecast SARIMAX (5 años)", linestyle="--")
plt.title("Pronóstico PM2.5 Carapungo (5 años)")
plt.ylabel("PM2.5 [µg/m³]")
plt.xlabel("Fecha")
plt.legend()
plt.tight_layout()
plt.show()


## FORECAST DIARIO

from sklearn.ensemble     import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import mean_squared_error, mean_absolute_error

# 2.1. hold‐out última ventana de 10 días para test
horizon = 10
X_train, X_test = X.iloc[:-horizon], X.iloc[-horizon:]
y_train, y_test = y.iloc[:-horizon], y.iloc[-horizon:]

# 2.2. Escalado
scaler = StandardScaler().fit(X_train)
X_tr = scaler.transform(X_train)
X_te = scaler.transform(X_test)

# 2.3. GridSearch con TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
param_grid = {
    "n_estimators": [100, 200],
    "max_depth":    [5, 10, None]
}
rf = RandomForestRegressor(random_state=0)
gscv = GridSearchCV(
    rf, param_grid,
    cv       = tscv,
    scoring  = "neg_mean_absolute_error",
    n_jobs   = -1
)
gscv.fit(X_tr, y_train)

best_rf = gscv.best_estimator_
print("Mejor RF params:", gscv.best_params_)

# 2.4. predecir en test y métricas
pred_rf = best_rf.predict(X_te)
rmse_rf = np.sqrt(mean_squared_error(y_test, pred_rf))
mae_rf  = mean_absolute_error(y_test, pred_rf)

print(f"RF Daily → RMSE: {rmse_rf:.2f}, MAE: {mae_rf:.2f}")

Mejor RF params: {'max_depth': 10, 'n_estimators': 100}
RF Daily → RMSE: 4.17, MAE: 3.04

plt.figure(figsize=(10,4))
plt.plot(y_test.index, y_test, label="Observado", marker="o")
plt.plot(y_test.index, pred_rf, label="Predicho RF", marker="x")
plt.title("Pronóstico diario de PM2.5 (Carapungo) – Random Forest (últimos 10 días)")
plt.ylabel("PM2.5 [µg/m³]")
plt.xlabel("Fecha")
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(10,4))
plt.plot(y_test.index, y_test, label="Observado", marker="o")
plt.plot(y_test.index, pred_rf, label="Predicho RF", marker="x")
plt.title("Pronóstico diario de PM2.5 (Carapungo) – Random Forest (últimos 10 días)")
plt.ylabel("PM2.5 [µg/m³]")
plt.xlabel("Fecha")
plt.legend()
plt.tight_layout()
plt.show()


# Supón que ya tienes el último día de tus datos:
last_day = fe_df.index[-1]

# Crea DataFrame de los próximos 10 días con las features que necesita el modelo
future_dates = pd.date_range(last_day + pd.Timedelta(days=1), periods=10, freq="D")

# Tomamos la última fila y generamos features para los próximos 10 días (forward-fill)
future_df = []
last_row = fe_df.iloc[-1].copy()
for i, d in enumerate(future_dates, 1):
	row = last_row.copy()
	row["month"] = d.month
	row["dayofyear"] = d.dayofyear
	row["dayofweek"] = d.dayofweek
	row["weekofyear"] = d.isocalendar().week
	row["sin_doy"] = np.sin(2 * np.pi * d.dayofyear / 365.0)
	row["cos_doy"] = np.cos(2 * np.pi * d.dayofyear / 365.0)
	# Actualiza lags: para lag1, usamos el último PM2.5 predicho o real
	for lag in (1, 2, 3, 7, 14):
		if lag == 1 and i > 1:
			row[f"PM2.5_lag{lag}"] = future_df[-1]["PM2.5"]
		else:
			row[f"PM2.5_lag{lag}"] = last_row["PM2.5"]
	# Rolling: mantenemos igual que el último día (aprox)
	# Changepoints
	for cp in ["2009-10-01", "2013-08-01", "2017-07-01"]:
		row[f"after_{cp}"] = int(d >= pd.to_datetime(cp))
	future_df.append(row)
	# Simula PM2.5 para el próximo día como el último valor (dummy, solo para lag)
	row["PM2.5"] = last_row["PM2.5"]

future_X = pd.DataFrame(future_df, index=future_dates)
future_X = future_X[X.columns]  # asegura el mismo orden y columnas

# Escalado
X_future_scaled = scaler.transform(future_X)

# Predicción
pred_rf_future = best_rf.predict(X_future_scaled)

# Luego graficas como antes:
plt.figure(figsize=(10,4))
plt.plot(future_dates, pred_rf_future, label="Forecast RF (10 días)", marker="x")
plt.title("Pronóstico diario PM2.5 (Carapungo) – Random Forest (10 días)")
plt.ylabel("PM2.5 [µg/m³]")
plt.xlabel("Fecha")
plt.legend()
plt.tight_layout()
plt.show()


#LSTM


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 3.1. crear secuencias (sliding window) para un paso adelante
def make_sequences(X, y, window=7):
    Xs, ys = [], []
    for i in range(window, len(X)):
        Xs.append(X[i-window:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

# 3.2. escala todo X antes de sequence
scaler = StandardScaler().fit(X)
X_all = scaler.transform(X)
y_all = y.values

# 3.3. secuencias y split 80/20
window = 7
X_seq, y_seq = make_sequences(X_all, y_all, window)
split = int(len(X_seq)*0.8)

X_tr, X_te = X_seq[:split], X_seq[split:]
y_tr, y_te = y_seq[:split], y_seq[split:]

# 3.4. definir modelo
model = Sequential([
    LSTM(64, input_shape=(window, X_seq.shape[2])),
    Dense(1)
])
model.compile(optimizer="adam", loss="mae")

# 3.5. train con early stopping
es = EarlyStopping(patience=5, restore_best_weights=True)
model.fit(
    X_tr, y_tr,
    validation_split=0.1,
    epochs=50,
    batch_size=32,
    callbacks=[es],
    verbose=2
)

# 3.6. predecir y métricas
pred_lstm = model.predict(X_te).ravel()
rmse_lstm = np.sqrt(mean_squared_error(y_te, pred_lstm))
mae_lstm  = mean_absolute_error(y_te, pred_lstm)

print(f"LSTM Daily → RMSE: {rmse_lstm:.2f}, MAE: {mae_lstm:.2f}")


80/80 - 2s - 19ms/step - loss: 10.7768 - val_loss: 5.7011
Epoch 2/50
80/80 - 0s - 3ms/step - loss: 5.1662 - val_loss: 4.8696
Epoch 3/50
80/80 - 0s - 3ms/step - loss: 5.1150 - val_loss: 4.9783
Epoch 4/50
80/80 - 0s - 3ms/step - loss: 5.0152 - val_loss: 5.2887
Epoch 5/50
80/80 - 0s - 3ms/step - loss: 4.7878 - val_loss: 5.3659
Epoch 6/50
80/80 - 0s - 3ms/step - loss: 4.6571 - val_loss: 5.8814
Epoch 7/50
80/80 - 0s - 3ms/step - loss: 4.5455 - val_loss: 6.7129
[1m23/23[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step
LSTM Daily → RMSE: 102.82, MAE: 38.75

plt.figure(figsize=(10,4))
plt.plot(y_te, label="Observado", marker="o")
plt.plot(pred_lstm, label="Predicho LSTM", marker="x")
plt.title("Pronóstico diario de PM2.5 (Carapungo) – LSTM (test set)")
plt.ylabel("PM2.5 [µg/m³]")
plt.xlabel("Índice test")
plt.legend()
plt.tight_layout()
plt.show()
