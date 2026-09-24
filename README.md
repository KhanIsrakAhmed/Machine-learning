# 📘 Final Paper Update

# Electricity Demand Forecasting Using TFT and Prophet Models

### A Comparative Study on Bangladesh Power Grid Data (2016–2024)

> **Course**: Machine Learning (20 Marks Term Paper)

---

## 📖 Our Story: How This Project Came Together

### Chapter 1: Why Did We Pick This Topic?

Bangladesh is a country of ~170 million people, and electricity is one of its biggest daily challenges. Every summer, people face **load shedding** (planned power cuts) because the grid cannot keep up with demand. Peak electricity demand has grown from ~5,000 MW in 2009 to over 15,000 MW in 2024.

We asked ourselves: **"Can we predict how much electricity Bangladesh will need, hour by hour?"**

If the power grid operators know tomorrow's demand accurately, they can:

- Turn on the right number of power plants (not too many, not too few)
- Avoid blackouts and load shedding
- Save fuel costs (running extra plants wastes money)
- Plan for future infrastructure

We found that existing studies on Bangladesh mostly used older methods like **ARIMA** (a statistical model from the 1970s) and basic neural networks. Nobody had compared modern deep learning (TFT) with modern statistical methods (Prophet) on Bangladesh data. That became our **research gap**.

### Chapter 2: What Models Did We Choose and Why?

We chose two models that represent two fundamentally different approaches:

**1. Temporal Fusion Transformer (TFT)** — The "Smart Deep Learning" Approach

- Invented by Google researchers (Lim et al., 2021)
- Uses **attention mechanism** (same idea behind ChatGPT) to find patterns
- Can automatically figure out which input features matter most
- Very powerful but needs a GPU and lots of training time

**2. Facebook Prophet** — The "Simple Statistical" Approach

- Created by Meta/Facebook (Taylor & Letham, 2018)
- Breaks the data into: Trend + Seasonality + External factors
- Easy to use, fast to train, runs on CPU
- Good for business forecasting

**Why compare these two?** Because in real life, a power company needs to decide: _"Should we invest in expensive deep learning infrastructure, or is a simpler model good enough?"_ Our paper answers this question.

### Chapter 3: Our Dataset

We found a dataset on **Kaggle** called `electricity-demand-bangladesh`. Here is what it contains:

| Column Name       | What It Means                           | Example Value | Range                |
| ----------------- | --------------------------------------- | ------------- | -------------------- |
| `Datetime`        | Date and hour of recording              | 1/1/2016 0:00 | Jan 2016 – Dec 2024  |
| `Temperature(2m)` | Air temperature at 2 meters height (°C) | 17.51         | ~10°C to ~40°C       |
| `Rhumadity(2m)`   | Relative humidity at 2 meters (%)       | 69.47         | ~20% to 100%         |
| `SPressure(kPa)`  | Atmospheric surface pressure            | 101.7         | ~99 to ~103 kPa      |
| `Demand(MW)`      | Electricity demand — **our target**     | 4211          | ~3,000 to ~16,000 MW |

**Key numbers:**

- **Total records**: 78,912 (hourly data, 9 years)
- **Period**: January 1, 2016 to December 31, 2024
- **Frequency**: One reading every hour (24 readings per day)

### Chapter 4: How We Cleaned the Data

**Step 1 — Check for Missing Values**: We checked all columns. Result: **zero missing values** ✓

**Step 2 — Remove Outliers using IQR Method**: Some readings were abnormally high or low (sensor errors, unusual events). We used the **Interquartile Range (IQR)** method:

```
Q1 = 25th percentile (lower quarter)
Q3 = 75th percentile (upper quarter)
IQR = Q3 - Q1
Lower limit = Q1 - 1.5 × IQR
Upper limit = Q3 + 1.5 × IQR
```

Any value below the lower limit or above the upper limit = **outlier**. We replaced outliers with the column's **mean value** (average). Total outliers found and fixed: **2,768 across all columns**.

> **Why IQR?** It's a non-parametric method — it doesn't assume the data follows any specific distribution (like normal/bell curve). It's robust and widely used.

**Step 3 — Feature Engineering**: We created 10 new features from the datetime column:

| New Feature           | What It Captures                  | Example                    |
| --------------------- | --------------------------------- | -------------------------- |
| `hour` (0–23)         | Time of day pattern               | Demand peaks at 7 PM       |
| `day_of_week` (0–6)   | Weekday vs weekend                | Factories closed on Friday |
| `day_of_month` (1–31) | Monthly salary cycle effects      | —                          |
| `day_of_year` (1–366) | Seasonal position                 | Summer = high demand       |
| `month` (1–12)        | Seasonal pattern                  | July = peak summer         |
| `quarter` (1–4)       | Broader seasonal grouping         | Q2–Q3 = hot season         |
| `year`                | Long-term growth trend            | Demand grows each year     |
| `week_of_year` (1–53) | Weekly position in year           | —                          |
| `Lagged_col`          | Demand 4 hours ago                | Recent demand pattern      |
| `Rolling_mean`        | Average demand over last 24 hours | Daily trend                |

> **Why lag and rolling features?** If demand was high 4 hours ago, it's likely still high now. The 24-hour rolling mean captures the overall daily trend, smoothing out short spikes.

After creating lag/rolling features, we **dropped 23 rows** that had NaN values (because you can't calculate "4 hours ago" for the first 4 hours, or "24-hour average" for the first 24 hours).

**Step 4 — Normalization (StandardScaler)**:

```
z = (x - mean) / standard_deviation
```

This transforms every feature to have **mean = 0** and **standard deviation = 1**. Neural networks train much better with normalized inputs because large numbers (like demand in thousands) don't dominate small numbers (like humidity percentages).

> **Important**: We fit the scaler on training data only, then applied the same transformation to test data. This prevents **data leakage** (test data should never influence training).

### Chapter 5: How We Split the Data

| Split        | Period              | Size                  | Purpose                              |
| ------------ | ------------------- | --------------------- | ------------------------------------ |
| **Training** | Jan 2016 – Nov 2024 | 78,122 records (~99%) | Model learns patterns                |
| **Testing**  | Dec 2024            | 744 records (~1%)     | We check if predictions are accurate |

> **Why only 1 month for testing?** In time series, you always test on the most recent data. With 9 years of training data, even 1 month (744 hours) gives meaningful evaluation. This is standard practice for long time series.

### Chapter 6: Training the Models

**TFT (Temporal Fusion Transformer) Setup:**

| Setting               | Value | Plain English                                            |
| --------------------- | ----- | -------------------------------------------------------- |
| `input_chunk_length`  | 48    | Model looks at past 48 hours (2 days) to predict         |
| `output_chunk_length` | 24    | Model predicts next 24 hours (1 day ahead)               |
| `hidden_size`         | 64    | Internal brain size of the model                         |
| `lstm_layers`         | 2     | Two layers of LSTM memory cells                          |
| `dropout`             | 0.1   | Randomly turns off 10% of neurons to prevent overfitting |
| `batch_size`          | 32    | Processes 32 samples at a time                           |
| `n_epochs`            | 5     | Goes through the entire dataset 5 times                  |

- Platform: **Kaggle** with **NVIDIA Tesla P100 GPU** (16 GB)
- Training time: **~1,238 seconds (~20 minutes)**
- Library used: **Darts** (by Unit8, published in JMLR 2022)

**Prophet Setup:**

| Setting                   | Value                           | Plain English                                     |
| ------------------------- | ------------------------------- | ------------------------------------------------- |
| `growth`                  | linear                          | Assumes demand grows in a straight line over time |
| `daily_seasonality`       | True                            | Captures hour-by-hour patterns (peak at evening)  |
| `weekly_seasonality`      | True                            | Captures weekday/weekend differences              |
| `yearly_seasonality`      | True                            | Captures summer/winter differences                |
| `changepoint_prior_scale` | 0.05                            | How flexible the trend line can bend              |
| `seasonality_prior_scale` | 10.0                            | How strong seasonal patterns can be               |
| Weather regressors        | Temperature, Humidity, Pressure | Extra inputs to help prediction                   |

- Training time: **~200 seconds (~3 minutes)**
- Runs on **CPU** (no GPU needed)

### Chapter 7: Our Results — What We Found

Here are our **actual experimental results**:

| Metric            | TFT           | Prophet     | Winner     | What It Means                                |
| ----------------- | ------------- | ----------- | ---------- | -------------------------------------------- |
| **MAPE**          | **2.52%**     | 9.79%       | ✅ TFT     | TFT's average prediction error is only 2.52% |
| **MSE**           | **80,433**    | 1,002,374   | ✅ TFT     | TFT has 12× less squared error               |
| **RMSE**          | **283.61 MW** | 1,001.19 MW | ✅ TFT     | TFT is off by ~284 MW on average             |
| **MAE**           | **211.19 MW** | 778.82 MW   | ✅ TFT     | TFT's typical error is ~211 MW               |
| **R² Score**      | **0.9291**    | 0.1179      | ✅ TFT     | TFT explains 93% of demand variation         |
| **Training Time** | 1,238.5 s     | **199.7 s** | ✅ Prophet | Prophet trains 6× faster                     |

### What These Numbers Mean in Simple Words:

- **TFT achieved 2.52% MAPE** — This means if the actual demand is 10,000 MW, TFT predicts between 9,748 and 10,252 MW. That's excellent! (Below 5% MAPE is considered very good for load forecasting.)
- **Prophet achieved 9.79% MAPE** — Still usable, but on 10,000 MW demand, the prediction could be off by ~980 MW. That's a lot of power plants to miscalculate.
- **TFT R² = 0.9291** — TFT explains 93% of why demand goes up and down. Only 7% is unexplained.
- **Prophet R² = 0.1179** — Prophet only explains 12% of demand variation. It missed most patterns.
- **But Prophet trains 6× faster** — If you need a quick rough estimate, Prophet is useful.

### Key Takeaway:

> **TFT clearly wins on accuracy.** It's worth the extra training time and GPU cost if you need reliable forecasts for grid management. Prophet is fine for a quick rough estimate but not for critical operations.

### Chapter 8: What Decisions Can We Make from These Results?

1. **For Bangladesh Power Grid**: Use TFT-based models for official demand forecasting — 2.5% error is acceptable for generation scheduling
2. **For smaller utilities with limited computing**: Prophet can serve as a quick baseline, but should not be the primary forecasting tool
3. **Weather data matters**: Temperature showed the strongest correlation with demand (people use more AC when it's hot)
4. **Deep learning > traditional statistics** for this problem: The nonlinear relationship between weather and demand is better captured by TFT's attention mechanism

### Chapter 9: Challenges We Faced

1. **GPU limitations**: TFT needs GPU. On Kaggle free tier, we only trained for 5 epochs. More epochs would likely improve results further
2. **No holiday data**: Bangladesh has many holidays (Eid, Pohela Boishakh) that affect demand. We didn't model these
3. **Single test month**: We only tested on December 2024. Testing on summer months might show different results
4. **Dataset column naming**: The humidity column was misspelled as `Rhumadity(2m)` in the original dataset — we had to use it as-is

### Chapter 10: Future Improvements

- Train TFT for more epochs (20–50 instead of 5)
- Add Bangladesh holiday calendar
- Test on multiple months across different seasons
- Try hybrid approach: combine TFT and Prophet predictions (ensemble)
- Add economic data (GDP, industrial output) as features
- Implement real-time updating (online learning)
- Try other models: N-BEATS, DeepAR, XGBoost for comparison

---

## 📊 Reference Papers — What Each Paper Contributed

| #   | Paper                                                                                | Authors (Year)                            | What It Gave Us                   | How We Used It                                 |
| --- | ------------------------------------------------------------------------------------ | ----------------------------------------- | --------------------------------- | ---------------------------------------------- |
| 1   | Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting | Lim et al. (2021)                         | The TFT model architecture        | Our primary deep learning model                |
| 2   | Forecasting at Scale                                                                 | Taylor & Letham (2018)                    | The Prophet model                 | Our statistical baseline model                 |
| 3   | Attention Is All You Need                                                            | Vaswani et al. (2017)                     | Transformer attention mechanism   | Foundation of TFT's attention layer            |
| 4   | Long Short-Term Memory                                                               | Hochreiter & Schmidhuber (1997)           | LSTM architecture                 | Used inside TFT's encoder-decoder              |
| 5   | Darts: User-Friendly Modern ML for Time Series                                       | Herzen et al. (2022)                      | Darts library                     | Python library we used for TFT                 |
| 6   | Time Series Analysis: Forecasting and Control                                        | Box et al. (2015)                         | ARIMA foundations                 | Background on traditional methods              |
| 7   | Probabilistic Electric Load Forecasting                                              | Hong & Fan (2016)                         | Load forecasting best practices   | Guided our evaluation metrics                  |
| 8   | XGBoost: A Scalable Tree Boosting System                                             | Chen & Guestrin (2016)                    | Gradient boosting methods         | Context for ML approaches in literature review |
| 9   | DeepAR: Probabilistic Forecasting                                                    | Salinas et al. (2020)                     | Autoregressive RNN forecasting    | Related work in deep learning for time series  |
| 10  | Forecasting: Principles and Practice                                                 | Hyndman & Athanasopoulos (2018)           | Time series fundamentals          | Theoretical background                         |
| 11  | BPDB Annual Report 2022–2023                                                         | Bangladesh Power Development Board (2023) | Bangladesh electricity statistics | Justified our problem statement                |
| 12  | Short-term Load Forecasting Using ARIMA for Bangladesh                               | Hossain & Mahmud (2018)                   | ARIMA on Bangladesh data          | Showed existing work used older methods        |
| 13  | Electricity Demand Forecasting Using ANN                                             | Islam et al. (2019)                       | Neural network on Bangladesh data | Showed gap: no TFT/Prophet comparison exists   |

---

## 📁 Project File Structure

```
ml-paper-final/
├── BangladeshData_2016_2024.csv          ← The raw dataset (78,912 rows)
├── paper.tex                              ← IEEE LaTeX paper (final submission)
├── latex/ml_paper.pdf                     ← Compiled PDF of the paper
├── README.md                              ← This file (you're reading it)
├── mid_README.md                          ← Mid-term README (earlier version)
├── colab/                                 ← Google Colab notebook cells
│   ├── cell1_setup.py                     ← Install libraries
│   ├── cell2_data_loading.py              ← Load data + initial plots
│   ├── cell3_preprocessing.py             ← Clean data + feature engineering
│   ├── cell4_train_test_split.py          ← Split + normalize
│   ├── cell5_tft_model.py                 ← Train TFT model (~20 min)
│   ├── cell6_prophet_model.py             ← Train Prophet model (~3 min)
│   ├── cell7_comparison.py                ← Compare both models
│   └── cell8_results_export.py            ← Export all results
├── kaggle/                                ← Kaggle notebook cells (same logic)
└── machine-learning-paper-.../            ← Output archive with results
    ├── image1_data_overview.png           ← Full 9-year time series plot
    ├── image2_outlier_analysis.png        ← Before/after outlier removal
    ├── image3_correlation_heatmap.png     ← Feature correlations
    ├── image4_tft_results.png             ← TFT predictions vs actual
    ├── image5_prophet_results.png         ← Prophet predictions vs actual
    ├── image6_prophet_components.png      ← Prophet's trend + seasonality breakdown
    ├── image7_comparison.png              ← Both models side by side
    ├── image8_error_distribution.png      ← Error histograms
    ├── image9_metrics_comparison.png      ← Bar chart of all metrics
    ├── image10_zoomed_comparison.png      ← Zoomed view (first 7 days)
    ├── metrics_comparison.csv             ← Final metrics table
    ├── tft_results.csv                    ← TFT hourly predictions
    ├── prophet_results.csv                ← Prophet hourly predictions
    └── dataset_summary.json               ← Dataset statistics
```

### Code Execution Order:

```
cell1 (install) → cell2 (load) → cell3 (clean) → cell4 (split) → cell5 (TFT) → cell6 (Prophet) → cell7 (compare) → cell8 (export)
```

---

## 🎯 Possible Viva / Teacher Questions & Answers

### Q1: What is your project about?

**A:** We compared two forecasting models — TFT (deep learning) and Prophet (statistical) — to predict hourly electricity demand in Bangladesh using 9 years of real power grid data with weather information. TFT won with 2.52% error vs Prophet's 9.79%.

### Q2: Why did you choose electricity demand forecasting?

**A:** Bangladesh faces frequent load shedding due to inaccurate demand prediction. If grid operators know tomorrow's demand accurately, they can plan how many power plants to run, saving costs and avoiding blackouts.

### Q3: What variables are in your dataset?

**A:** Five columns: `Datetime` (hourly timestamp), `Temperature(2m)` (air temperature in °C), `Rhumadity(2m)` (humidity in %), `SPressure(kPa)` (atmospheric pressure), and `Demand(MW)` (our target — electricity demand in megawatts).

### Q4: How did you clean the data?

**A:** Three steps: (1) Checked for missing values — found none. (2) Removed outliers using the IQR method — found and replaced 2,768 outliers with column means. (3) Created 10 new features from the datetime (hour, month, day_of_week, etc.) plus a 4-hour lag and 24-hour rolling average.

### Q5: What is the IQR method?

**A:** IQR = Q3 − Q1 (difference between 75th and 25th percentile). Any value below Q1 − 1.5×IQR or above Q3 + 1.5×IQR is an outlier. It's non-parametric — doesn't assume any distribution shape.

### Q6: Why did you replace outliers with the mean instead of removing them?

**A:** Removing rows would create gaps in our time series, breaking the hourly continuity. Replacing with mean preserves the sequence structure while neutralizing extreme values.

### Q7: What is TFT and how does it work?

**A:** Temporal Fusion Transformer is a deep learning model with 4 main parts: (1) Variable Selection Networks — automatically picks the most important input features. (2) Gated Residual Networks — processes features with skip connections. (3) LSTM encoder-decoder — captures short-term patterns. (4) Multi-Head Attention — captures long-range patterns (like how demand today relates to same day last week).

### Q8: What is Prophet and how does it work?

**A:** Prophet decomposes time series into: `y(t) = g(t) + s(t) + h(t) + ε(t)` where g(t) = trend (overall direction), s(t) = seasonality (repeating patterns modeled with Fourier series — sum of sine/cosine waves), h(t) = external effects (weather regressors in our case), ε(t) = random noise.

### Q9: What is the attention mechanism?

**A:** Attention lets the model "focus" on the most relevant past time steps. Formula: `Attention(Q,K,V) = softmax(QKᵀ/√dₖ)V`. Think of it like this: when predicting 6 PM demand, the model pays more attention to 6 PM yesterday and last week, rather than 3 AM yesterday.

### Q10: What is StandardScaler normalization? Why is it needed?

**A:** Formula: `z = (x − mean) / std_dev`. It makes all features have mean=0 and std=1. Without this, demand values (3000–16000) would dominate temperature values (10–40) during training, causing the neural network to learn poorly.

### Q11: Why did you use a 4-hour lag feature?

**A:** If demand was high 4 hours ago, it's likely still elevated now (industrial processes don't stop instantly, weather doesn't change suddenly). The lag gives the model "memory" of recent demand without using the actual future values.

### Q12: What is R² score and what does yours mean?

**A:** R² measures what proportion of the target's variation is explained by the model. R²=1 is perfect, R²=0 means no better than guessing the average. Our TFT got R²=0.9291 (explains 93% of variation — very good). Prophet got R²=0.1179 (only 12% — poor).

### Q13: What is MAPE and why is it the most important metric here?

**A:** MAPE = average of |actual − predicted| / actual × 100%. It gives percentage error, which is intuitive: "our model is off by X%". For load forecasting, MAPE < 5% is considered excellent. Our TFT achieved 2.52% — well within the excellent range.

### Q14: Why does temperature affect electricity demand?

**A:** Bangladesh has a tropical climate. When temperature rises, people turn on air conditioners and fans. Our correlation analysis showed temperature is the strongest predictor of demand. Summer months (May–September) show demand peaks of 15,000+ MW versus winter (December–February) at ~8,000–10,000 MW.

### Q15: What is a covariate in time series?

**A:** A covariate is an additional variable (besides the target) that helps predict the target. Our covariates are weather variables (temperature, humidity, pressure). For TFT, they're called "future covariates" (because we can get weather forecasts ahead of time). For Prophet, they're called "regressors."

### Q16: Why is Prophet so much worse than TFT?

**A:** Prophet uses a simple additive model (trend + seasonality + regressors). It can't capture complex **nonlinear interactions** between features. For example, the relationship between temperature and demand isn't linear — demand spikes sharply above 30°C due to AC usage. TFT's attention mechanism and neural network layers can learn these nonlinear patterns.

### Q17: If TFT is better, why would anyone use Prophet?

**A:** Prophet trains in 3 minutes on CPU vs TFT's 20 minutes on GPU. If you need a quick estimate, Prophet works. Also, Prophet gives clear decomposition (you can see the trend and seasonality separately), making it easier to explain to non-technical stakeholders. For initial exploration or when computational resources are limited, Prophet is still valuable.

### Q18: What is the train/test split ratio?

**A:** Training: Jan 2016 to Nov 2024 (78,122 records, ~99%). Testing: Dec 2024 (744 records, ~1%). In time series, we always test on the most recent period to simulate real-world forecasting.

### Q19: What is overfitting and how did you prevent it?

**A:** Overfitting = model memorizes training data but fails on new data. We prevented it by: (1) Using dropout=0.1 in TFT (randomly disables 10% of neurons during training), (2) Only training for 5 epochs (not too many passes), (3) Using StandardScaler normalization, (4) Having a large training set (78,000+ records).

### Q20: What are the limitations of your study?

**A:** (1) Test period is only December — a winter month, so we haven't tested on summer. (2) TFT trained for only 5 epochs due to time constraints. (3) No holiday modeling (Eid, Pohela Boishakh). (4) Only national-level data — no regional breakdown. (5) Limited hyperparameter tuning.

### Q21: What is feature engineering?

**A:** Creating new informative features from raw data. We extracted temporal features (hour, month, day_of_week) from the datetime column, created a lag feature (demand 4 hours ago), and a rolling mean (24-hour average). These help the model understand periodic patterns that a raw datetime string cannot directly convey.

### Q22: What is Fourier series in Prophet?

**A:** Fourier series represents periodic (repeating) patterns as sums of sine and cosine waves: `s(t) = Σ(aₙ·cos(2πnt/P) + bₙ·sin(2πnt/P))`. Prophet uses this to model daily seasonality (P=1 day), weekly seasonality (P=7 days), and yearly seasonality (P=365.25 days). The model learns the coefficients (aₙ, bₙ) from data.

### Q23: What tools and libraries did you use?

**A:** Python 3.10, Darts 0.36 (for TFT), Prophet 1.1, PyTorch 2.6 (backend for TFT), scikit-learn 1.7 (for StandardScaler), pandas (data manipulation), numpy (math), matplotlib (plots). Platform: Kaggle with GPU.

### Q24: What is LSTM and why is it inside TFT?

**A:** LSTM (Long Short-Term Memory) is a type of recurrent neural network that can remember information over long sequences. Inside TFT, the LSTM encoder reads the past 48 hours and creates a summary, then the LSTM decoder uses that summary to generate predictions. LSTMs have "gates" that control what to remember and what to forget.

### Q25: Could you improve the results further?

**A:** Yes: (1) Train TFT for 20–50 epochs instead of 5. (2) Try hyperparameter tuning (different hidden sizes, learning rates). (3) Add holiday calendar for Bangladesh. (4) Ensemble TFT and Prophet. (5) Use more weather features (wind speed, solar radiation). (6) Apply transfer learning from other countries' data.

### Q26: What is the difference between MSE, RMSE, and MAE?

**A:** MSE = average of (error²) — penalizes large errors more because of squaring. RMSE = √MSE — same units as target (MW), easier to interpret. MAE = average of |error| — treats all errors equally. RMSE > MAE means there are some large outlier errors (squared penalty amplifies them).

### Q27: What is residual analysis?

**A:** Residual = Actual − Predicted. A good model should have residuals that are: (1) Centered around zero (no systematic bias), (2) Randomly distributed (no patterns), (3) Similar spread across all predictions (homoscedastic). We generated error distribution histograms to check this.

### Q28: What does the correlation heatmap show?

**A:** It shows how strongly each pair of variables is related. Values range from -1 (perfectly opposite) to +1 (perfectly together). We found: Temperature ↔ Demand has strong positive correlation (hot = more demand). Humidity ↔ Temperature has negative correlation (humid days tend to be cooler, or vice versa).

### Q29: What is changepoint detection in Prophet?

**A:** Changepoints are moments where the overall trend changes direction. For example, if demand was growing at 5% per year and suddenly starts growing at 10%, that's a changepoint. Prophet automatically places potential changepoints at regular intervals and uses a Laplacian prior to select only the significant ones. `changepoint_prior_scale=0.05` controls how flexible this is.

### Q30: How do you reproduce this experiment?

**A:** (1) Upload `BangladeshData_2016_2024.csv` to Google Drive. (2) Open Google Colab. (3) Enable GPU runtime. (4) Run cells 1–8 in order from the `colab/` folder. (5) After cell 1, restart session, then run cells 2–8. Total time: ~25 minutes. All figures and results auto-save to Drive.

---

## 📎 Full References

1. Lim, B., Arik, S.O., Loeff, N., & Pfister, T. (2021). Temporal Fusion Transformers for interpretable multi-horizon time series forecasting. _International Journal of Forecasting_, 37(4), 1748–1764.
2. Taylor, S.J. & Letham, B. (2018). Forecasting at scale. _The American Statistician_, 72(1), 37–45.
3. Vaswani, A., et al. (2017). Attention is all you need. _NeurIPS_, 5998–6008.
4. Hochreiter, S. & Schmidhuber, J. (1997). Long short-term memory. _Neural Computation_, 9(8), 1735–1780.
5. Herzen, J., et al. (2022). Darts: User-friendly modern machine learning for time series. _JMLR_, 23(124), 1–6.
6. Box, G.E.P., et al. (2015). _Time Series Analysis: Forecasting and Control_, 5th ed. Wiley.
7. Hong, T. & Fan, S. (2016). Probabilistic electric load forecasting. _International Journal of Forecasting_, 32(3), 914–938.
8. Chen, T. & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. _ACM SIGKDD_, 785–794.
9. Salinas, D., et al. (2020). DeepAR: Probabilistic forecasting with autoregressive recurrent networks. _IJF_, 36(3), 1181–1191.
10. Hyndman, R.J. & Athanasopoulos, G. (2018). _Forecasting: Principles and Practice_, 2nd ed. OTexts.
11. Bangladesh Power Development Board (2023). Annual Report 2022–2023. BPDB, Dhaka.
12. Hossain, M.S. & Mahmud, H. (2018). Short-term load forecasting using ARIMA model for Bangladesh. _International Journal of Energy_, 7(2), 31–40.
13. Islam, M.A., et al. (2019). Electricity demand forecasting of Bangladesh using artificial neural network. _Journal of Electrical Engineering_, 47(1), 1–8.

---

_This README was prepared as the final update document for the Machine Learning term paper project._
