---
layout: default
title: Feature engineering temporali
parent: Teoria
nav_order: 3
math: mathjax
description: >-
  Decomposizione del timestamp, feature di stagionalità, distanza
  geografica Haversine, aggregati expanding per cliente. Costruite con
  attenzione al data leakage temporale.
---

# Feature engineering temporali per fraud detection
{: .no_toc }

## Indice
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## 1. Perché il feature engineering è il cuore di fraud detection

I dataset di transazioni come quello Kaggle hanno feature grezze poco informative se prese singolarmente: `amt`, `lat/long`, `category`. La performance reale dipende da **come queste vengono combinate** in feature derivate che catturano i pattern di frode.

Tre famiglie di feature derivate sono particolarmente efficaci:

1. **Temporali**: pattern legati al ciclo orario, settimanale, stagionale.
2. **Geografiche**: distanza fra cliente e merchant.
3. **Aggregati per cliente**: contesto storico — la transazione corrente è normale per quel cliente?

Nel modulo `fraud_pipeline.features`, tutte e tre sono implementate dentro un `BaseEstimator + TransformerMixin` (`FraudFeatureEngineer`) per essere componibili in `sklearn.pipeline.Pipeline`.

## 2. Feature temporali da `trans_date_trans_time`

Il timestamp grezzo (es. `2020-06-15 14:32:00`) non è utile a un modello: è una stringa o un timestamp Unix. Decomponiamolo in componenti atomiche.

```python
ts = pd.to_datetime(df["trans_date_trans_time"])
df["hour"]        = ts.dt.hour          # 0..23
df["day_of_week"] = ts.dt.dayofweek     # 0..6 (Mon=0)
df["month"]       = ts.dt.month         # 1..12
df["is_weekend"]  = (ts.dt.dayofweek >= 5).astype(int)
df["is_night"]    = ((ts.dt.hour < 6) | (ts.dt.hour >= 22)).astype(int)
```

### Razionale

- **`hour`**: le frodi tendono a concentrarsi di **notte** (carte rubate usate quando il proprietario dorme).
- **`is_night`**: feature derivata booleana, redundante con `hour` ma utile per modelli lineari che faticano a modellare la non monotonia (es. picco a 02:00 + valle alle 09:00).
- **`day_of_week`**: alcuni schemi di frode sfruttano weekend (call center bancari ridotti).
- **`month`**: stagionalità (es. picchi di frodi durante shopping season).

### Codifica ciclica (alternativa)

`hour=23` e `hour=0` sono distanti 1 ora ma il modello li vede come 23. Una codifica ciclica risolve:

$$
\text{hour\_sin} = \sin\left(2\pi \cdot \frac{\text{hour}}{24}\right), \qquad
\text{hour\_cos} = \cos\left(2\pi \cdot \frac{\text{hour}}{24}\right)
$$

Per modelli lineari aiuta. Per Random Forest / XGBoost è meno necessario perché gli alberi gestiscono naturalmente la non-monotonia.

## 3. Età del cliente al momento della transazione

`dob` (date of birth) è una stringa. Convertita in età, diventa una feature potente: gli schemi di frode possono variare con l'età (clienti anziani sono target più frequenti di scam).

```python
dob = pd.to_datetime(df["dob"])
age_days = (ts - dob).dt.days
df["customer_age_years"] = (age_days / 365.25).clip(lower=0, upper=120)
```

Il `clip` difende da date di nascita errate (es. `dob` futura → età negativa).

## 4. Distanza geografica: la formula di Haversine

Il dataset contiene `(lat, long)` cliente e `(merch_lat, merch_long)` merchant. La distanza euclidea sui gradi è sbagliata (il meridiano e l'equatore non hanno la stessa "lunghezza" per grado).

La **formula di Haversine** restituisce la distanza geodetica su una sfera:

$$
d = 2R \cdot \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta\varphi}{2}\right) + \cos\varphi_1 \cos\varphi_2 \sin^2\left(\frac{\Delta\lambda}{2}\right)}\right)
$$

dove $R = 6371$ km (raggio terrestre), $\varphi$ = latitudine, $\lambda$ = longitudine, $\Delta\varphi = \varphi_2 - \varphi_1$.

Implementazione vettorizzata in `features.py`:

```python
def haversine_km(lat1, lon1, lat2, lon2):
    lat1_r, lon1_r, lat2_r, lon2_r = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2_r - lat1_r, lon2_r - lon1_r
    a = np.sin(dlat/2)**2 + np.cos(lat1_r)*np.cos(lat2_r)*np.sin(dlon/2)**2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
```

### Feature derivate

- `distance_km`: distanza in km. Distribuzione molto skewed verso 0 (la maggioranza delle tx è a pochi km dal cliente).
- `is_far_tx`: flag booleano `distance_km > 500`. Una transazione molto lontana è statisticamente più sospetta (carta rubata usata in una città diversa, o card-not-present con localizzazione anomala).

## 5. Aggregati per cliente: il contesto storico

Le feature più predittive in fraud detection sono spesso **relazionali**: "quanto la transazione corrente è anomala rispetto al cliente?". Una transazione di $200 è normale per chi spende mediamente $250 ma sospetta per chi spende mediamente $20.

### Z-score vs storico

$$
z_t = \frac{\text{amt}_t - \mu_{<t}}{\sigma_{<t} + \epsilon}
$$

dove $\mu_{<t}$ e $\sigma_{<t}$ sono media e deviazione standard delle transazioni precedenti dello stesso cliente. Un z-score elevato (>3) segnala una transazione anomala.

### CRITICO: no-leakage temporale

L'errore più comune nel calcolo degli aggregati è usare `.groupby().transform("mean")` su tutto il dataset. Questo include il **futuro** nella media e gonfia artificialmente le metriche.

**Sbagliato** (leakage):

```python
df["customer_mean_amt"] = df.groupby("cc_num")["amt"].transform("mean")
```

**Corretto** (no leakage):

```python
df = df.sort_values(["cc_num", "trans_date_trans_time"])
prev_amt = df.groupby("cc_num")["amt"].shift(1)
df["customer_mean_amt_so_far"] = (
    prev_amt.groupby(df["cc_num"]).expanding().mean().reset_index(0, drop=True)
)
```

Lo `shift(1)` PRIMA dell'expanding garantisce che la riga corrente non entri nella sua stessa media. La riga $t$ vede solo statistiche calcolate su $0, 1, \dots, t-1$.

!!! warning "Test di no-leakage"
    Nel file `tests/test_features.py` c'è un test esplicito: `test_customer_aggregates_no_future_leakage`. Verifica che la **prima** transazione di ogni cliente abbia `tx_count_so_far == 0`. Se non fosse così, il transformer starebbe includendo la riga corrente nel suo aggregato.

### Implementazione efficiente

`expanding().mean()` su 1,5M righe con 1.000 gruppi è lento. La nostra implementazione usa cumulative sums vettorizzati:

```python
cum_count_prev = grp.cumcount()
cum_sum_prev   = grp.shift(1).groupby(cc_num).cumsum()
mean_prev      = cum_sum_prev / cum_count_prev
```

Tempi ~5–10 secondi su 1,5M righe vs ~30s con `expanding()`.

## 6. Trasformazioni dell'importo

Il campo `amt` è skewed (long tail di importi alti). Trasformazioni utili:

- **`log_amt = log(1 + amt)`**: comprime la coda; rende la distribuzione più gaussiana; `log1p` gestisce $amt = 0$.
- **`is_small_amt = (amt < 1)`**: flag binario. Importi < $1 sono spesso transazioni di **test**: i fraudster verificano che la carta funzioni con una transazione minima prima di colpire con un importo grosso.

## 7. Categorie e merchant: encoding

Il dataset ha:

- `category`: 14 categorie merchant. **OneHotEncoder** standard.
- `merchant`: 693 merchant unici. OneHot diretto produrrebbe 693 colonne dummy → curse of dimensionality. Strategie:
    - **Drop**: rimuovere la colonna. La perdita di segnale è limitata se si tengono `category` e `distance_km`.
    - **Target encoding** (con prior): sostituire ogni merchant con la sua probabilità storica di frode. Richiede attenzione al leakage (calcolare il prior solo sul train) e regularization (Bayesian shrinkage).
    - **Hashing trick**: `HashingVectorizer` su una dimensione fissa (es. 64).

In questa pipeline droppiamo `merchant` per default (vedi `preprocessing.HIGH_CARDINALITY_COLUMNS`). Il target encoding è discusso nel file `docs/scelte_tecniche/scelte_modello.md`.

## 8. Feature engineer come `Pipeline` step

`FraudFeatureEngineer` implementa `fit/transform` ed è inserita come **primo step** della pipeline:

```python
from sklearn.pipeline import Pipeline
pipeline = Pipeline(steps=[
    ("feature_engineer", FraudFeatureEngineer()),
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(...)),
])
```

Vantaggi:

1. **No leakage in CV**: la trasformazione viene fatta dentro ogni fold sul subset di training.
2. **Serializzabile con joblib**: l'inferenza usa la stessa logica del training, bit per bit.
3. **Testabile**: ogni transformer ha una signature standard e si testa in isolamento.

## 9. Riferimenti

- **Bahnsen, A. C., Aouada, D., Stojanovic, A., Ottersten, B.** (2016). *Feature engineering strategies for credit card fraud detection*, Expert Systems with Applications 51.
- **Pozzolo, A. D., Caelen, O., Le Borgne, Y., Waterschoot, S., Bontempi, G.** (2014). *Learned lessons in credit card fraud detection from a practitioner perspective*, ESWA 41(10).
- **Sklearn user guide**: [Pipeline & FeatureUnion](https://scikit-learn.org/stable/modules/compose.html).
- **scikit-lego** (opzionale): [GroupedTransformer](https://scikit-lego.netlify.app/) per aggregati per gruppo dentro `Pipeline`.
