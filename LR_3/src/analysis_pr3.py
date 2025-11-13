import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats



plt.rcParams['figure.figsize'] = (8,5)
pd.set_option("display.max_columns", 200)

#для вибору файлу в Google Colab
try:
    from google.colab import files
    import io
    IN_COLAB = True
except ImportError:
    IN_COLAB = False




if IN_COLAB:
    print("Оберіть файл (.csv або .xlsx) для завантаження")
    uploaded = files.upload()

    if len(uploaded) == 0:
        raise RuntimeError("Файл не було обрано.")

    filename = list(uploaded.keys())[0]
    print("Використовую файл:", filename)

    if filename.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(uploaded[filename]))
    else:
        df = pd.read_csv(io.BytesIO(uploaded[filename]))
else:
    #можна задати свій шлях
    path = os.path.normpath("data.csv")   
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл не найден: {path}")
    print("Використовую файл:", path)
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, sheet_name=0)
    else:
        df = pd.read_csv(path)

print("Розмірність:", df.shape)
display(df.head())

print("Колонки у файлі:", list(df.columns))
df["Definition_len"] = df["Definition"].astype(str).str.len()
TARGET_COLUMN = "Definition_len"

y = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
print("TARGET_COLUMN =", TARGET_COLUMN)

X_raw = df.drop(columns=[TARGET_COLUMN]).copy()


for c in X_raw.columns:
    if X_raw[c].dtype == "object":
        X_raw[c] = X_raw[c].astype(str).str.replace(",", ".", regex=False)
    X_raw[c] = pd.to_numeric(X_raw[c], errors="ignore")


X = X_raw.select_dtypes(include=[np.number]).copy()

for c in X.columns:
    if X[c].isna().any():
        X[c] = X[c].fillna(X[c].median())


if y.isna().any():
    mask = ~y.isna()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)


const_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
if const_cols:
    X = X.drop(columns=const_cols)

print("Фінальна кількість ознак:", X.shape[1])
print("Розмір X:", X.shape, "| Розмір y:", y.shape)


if X.shape[1] > 0:
    metric_for_normality = X.columns[0]
    x = X[metric_for_normality].values
    N = len(x)
    mean = np.mean(x); var = np.var(x, ddof=1); std = np.sqrt(var)
    print(f"Метрика: {metric_for_normality} | N={N} | mean={mean:.4f} | var={var:.4f} | std={std:.4f}")

    plt.figure()
    plt.hist(x, bins=30, density=True, alpha=0.6)
    xs = np.linspace(x.min(), x.max(), 200)
    pdf = (1/(std*np.sqrt(2*np.pi))) * np.exp(-0.5*((xs-mean)/std)**2) if std>0 else np.zeros_like(xs)
    plt.plot(xs, pdf)
    plt.title(f"Гістограма та нормальний PDF: {metric_for_normality}")
    plt.show()

    plt.figure()
    stats.probplot(x, dist="norm", plot=plt)
    plt.title(f"QQ-plot: {metric_for_normality}")
    plt.show()

    if std > 0:
        ks_stat, ks_p = stats.kstest((x-mean)/std, 'norm')
        print(f"KS-stat={ks_stat:.4f}, p-value={ks_p:.4f} (H0: нормальність)")
else:
    print("Немає числових ознак для демонстрації нормальності.")


corr = X.join(y.rename('TARGET')).corr()
print(corr['TARGET'].sort_values(ascending=False).head(15))

plt.figure()
plt.imshow(corr.values, aspect='auto', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.index)), corr.index)
plt.title("Матриця кореляцій (X та TARGET)")
plt.tight_layout()
plt.show()


if X.shape[1] == 0:
    raise ValueError("Немає придатних числових ознак для моделювання.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print("Train shape:", X_train.shape, "| Test shape:", X_test.shape)

if X_train.shape[1] >= 2:
    X_train_sc_df = pd.DataFrame(X_train_sc, columns=X_train.columns)
    keep_cols = [c for c in X_train_sc_df.columns if X_train_sc_df[c].std(ddof=0) > 0]
    X_train_sc_df = X_train_sc_df[keep_cols]
    vif = pd.Series(
        [variance_inflation_factor(X_train_sc_df.values, i)
         for i in range(X_train_sc_df.shape[1])],
        index=X_train_sc_df.columns, name="VIF"
    ).sort_values(ascending=False)
    print("VIF (високі значення ⇒ мультиколінеарність):")
    print(vif.head(15).to_string())
else:
    print("VIF пропущено (менше 2 ознак).")



lin = LinearRegression()
lin.fit(X_train_sc, y_train)
y_pred = lin.predict(X_test_sc)

r2 = r2_score(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, y_pred)

eps = 1e-12
mre = np.abs(y_test - y_pred) / (np.abs(y_test) + eps)
mmre = float(np.mean(mre))
pred25 = float(np.mean(mre < 0.25))

print(f"R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, MMRE={mmre:.4f}, PRED(0.25)={pred25:.4f}")

X_train_ols = sm.add_constant(X_train_sc)
ols = sm.OLS(y_train, X_train_ols).fit()
print(ols.summary())

residuals = y_test - y_pred

plt.figure()
plt.hist(residuals, bins=30, density=True, alpha=0.7)
plt.title("Гістограма залишків")
plt.show()
plt.figure()
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("QQ-plot залишків")
plt.show()

from scipy.stats import shapiro
sh_stat, sh_p = shapiro(residuals.sample(min(5000, len(residuals)), random_state=42))
print(f"Shapiro-Wilk: stat={sh_stat:.4f}, p={sh_p:.4f} (H0: нормальність)")

thr = 3*np.std(residuals)
outliers_idx = np.where(np.abs(residuals - np.mean(residuals)) > thr)[0]
print(f"Потенційні викиди (|resid|>3σ): {len(outliers_idx)} з {len(residuals)}")


kf = KFold(n_splits=5, shuffle=True, random_state=42)

def r2_cv(X_df, y_s):
    scores = []
    for tr, te in kf.split(X_df):
        scaler_cv = StandardScaler()
        X_tr = scaler_cv.fit_transform(X_df.iloc[tr])
        X_te = scaler_cv.transform(X_df.iloc[te])
        m = LinearRegression().fit(X_tr, y_s.iloc[tr])
        yp = m.predict(X_te)
        scores.append(r2_score(y_s.iloc[te], yp))
    return np.array(scores)

scores = r2_cv(X, y)
print("CV R² (5-fold): mean=", scores.mean().round(4), " std=", scores.std().round(4))


import os
os.makedirs("results", exist_ok=True)
report_path = os.path.join("results", "PR3_report_export.md")

lines = []
lines.append("# Звіт ПР №3 (експорт)\n")
lines.append(f"- Кількість ознак: {X.shape[1]}\n")
lines.append(f"- Train/Test розмір: {X_train.shape} / {X_test.shape}\n")
lines.append(f"- R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, MMRE={mmre:.4f}, PRED(0.25)={pred25:.4f}\n")

with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("Збережено:", report_path)
if IN_COLAB:
    files.download(report_path)