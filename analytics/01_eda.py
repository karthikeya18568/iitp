from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / 'data'
OUT_DIR = BASE / 'outputs'
OUT_DIR.mkdir(exist_ok=True)

# The assignment requires the raw Titanic dataset to be loaded once and immediately
# saved as an offline fallback. In a normal connected run this uses seaborn's loader.
# If the network is unavailable, the committed titanic.csv is used instead.
def load_once():
    fallback = BASE / 'titanic.csv'
    try:
        import seaborn as sns
        df = sns.load_dataset('titanic')
        df.to_csv(fallback, index=False)
        return df, 'sns.load_dataset(\'titanic\')'
    except Exception as exc:
        df = pd.read_csv(fallback)
        return df, f'pd.read_csv fallback ({type(exc).__name__})'


def pct_missing(df):
    return (df.isna().mean() * 100).loc[lambda s: s > 0]


def iqr_outliers(series):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (series < lo) | (series > hi)
    return int(mask.sum()), float(lo), float(hi)


def savefig(name):
    plt.tight_layout()
    plt.savefig(OUT_DIR / name, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    df, source = load_once()
    raw_shape = df.shape
    missing = pct_missing(df)
    import io
    raw_buf = io.StringIO(); df.info(buf=raw_buf); raw_info = raw_buf.getvalue()
    raw_desc = df.describe(include='all').transpose().to_string()

    # Cleaning according to the required threshold rule.
    clean = df.copy()
    # age: 19.87% -> 5%-30%, median imputation.
    age_median = clean['age'].median()
    clean['age'] = clean['age'].fillna(age_median)
    # embarked: 0.22% -> under 5%, drop affected rows.
    clean = clean.dropna(subset=['embarked']).reset_index(drop=True)
    # deck: 77.10% -> too high for reliable imputation; drop the column and justify it.
    clean = clean.drop(columns=['deck'])
    # embark_town duplicates embarked; keep it for descriptive use, but modeling uses embarked.

    clean_path = DATA_DIR / 'titanic_clean.csv'
    clean.to_csv(clean_path, index=False)

    # Profiling
    info_lines = []
    buf = io.StringIO(); clean.info(buf=buf); info_lines = buf.getvalue()
    desc = clean.describe(include='all').transpose().to_string()

    # Univariate statistics
    age_out, age_lo, age_hi = iqr_outliers(clean['age'])
    fare_out, fare_lo, fare_hi = iqr_outliers(clean['fare'])
    fare_mean = clean['fare'].mean(); fare_median = clean['fare'].median(); fare_mode = clean['fare'].mode().iloc[0]
    skew = 'right-skewed' if fare_mean > fare_median > fare_mode else ('left-skewed' if fare_mean < fare_median < fare_mode else 'not strictly ordered')

    # Charts: age and fare distributions.
    plt.figure(figsize=(7, 4)); plt.hist(clean['age'], bins=20); plt.title('Age Distribution'); plt.xlabel('Age'); plt.ylabel('Count'); savefig('01_age_histogram.png')
    plt.figure(figsize=(7, 4)); plt.boxplot(clean['age'], vert=False); plt.title('Age Box Plot'); plt.xlabel('Age'); savefig('02_age_boxplot.png')
    plt.figure(figsize=(7, 4)); plt.hist(clean['fare'], bins=30); plt.title('Fare Distribution'); plt.xlabel('Fare'); plt.ylabel('Count'); savefig('03_fare_histogram.png')
    plt.figure(figsize=(7, 4)); plt.boxplot(clean['fare'], vert=False); plt.title('Fare Box Plot'); plt.xlabel('Fare'); savefig('04_fare_boxplot.png')

    # Bivariate survival rates using boolean masks.
    sex_rates = clean.groupby('sex', as_index=False)['survived'].mean().rename(columns={'survived':'survival_rate'})
    class_rates = clean.groupby('pclass', as_index=False)['survived'].mean().rename(columns={'survived':'survival_rate'})
    sex_class_rates = clean.groupby(['sex','pclass'], as_index=False)['survived'].mean().rename(columns={'survived':'survival_rate'})

    # Explicit boolean-mask examples requested by the brief.
    female_rate = clean.loc[clean['sex'].eq('female'), 'survived'].mean()
    male_rate = clean.loc[clean['sex'].eq('male'), 'survived'].mean()
    first_rate = clean.loc[clean['pclass'].eq(1), 'survived'].mean()
    second_rate = clean.loc[clean['pclass'].eq(2), 'survived'].mean()
    third_rate = clean.loc[clean['pclass'].eq(3), 'survived'].mean()
    female_first_rate = clean.loc[(clean['sex'].eq('female')) & (clean['pclass'].eq(1)), 'survived'].mean()
    male_third_rate = clean.loc[(clean['sex'].eq('male')) & (clean['pclass'].eq(3)), 'survived'].mean()

    corr_cols = ['survived','pclass','age','sibsp','parch','fare']
    corr = clean[corr_cols].corr()
    plt.figure(figsize=(8, 6)); sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True); plt.title('Titanic Numeric Correlation Matrix'); savefig('05_correlation_heatmap.png')
    pairs = []
    for i, a in enumerate(corr_cols):
        for b in corr_cols[i+1:]:
            pairs.append((a,b,float(corr.loc[a,b]),abs(float(corr.loc[a,b]))))
    top2 = sorted(pairs, key=lambda x: x[3], reverse=True)[:2]

    # Four multivariate charts.
    plt.figure(figsize=(7, 4))
    for sex in ['female','male']:
        sub=sex_class_rates[sex_class_rates['sex']==sex]
        plt.plot(sub['pclass'], sub['survival_rate'], marker='o', label=sex)
    plt.ylim(0,1); plt.xticks([1,2,3]); plt.title('Survival Rate by Sex and Passenger Class'); plt.xlabel('Passenger class'); plt.ylabel('Survival rate'); plt.legend(); savefig('06_survival_sex_class.png')
    plt.figure(figsize=(7, 4)); clean.boxplot(column='fare', by='survived'); plt.suptitle(''); plt.title('Fare by Survival'); plt.xlabel('Survived (0=No, 1=Yes)'); savefig('07_fare_by_survival.png')
    plt.figure(figsize=(7, 4))
    for survived in [0,1]:
        sub=clean[clean['survived']==survived]
        plt.scatter(sub['age'], sub['fare'], alpha=0.5, label=f'survived={survived}')
    plt.title('Age vs Fare by Survival'); plt.xlabel('Age'); plt.ylabel('Fare'); plt.legend(); savefig('08_age_fare_survival.png')
    pivot=clean.groupby(['pclass','embarked'])['survived'].mean().unstack(); ax=pivot.plot(kind='bar',figsize=(7,4)); ax.set_ylim(0,1); ax.set_title('Survival by Class and Embarkation'); ax.set_ylabel('Survival rate'); plt.tight_layout(); plt.savefig(OUT_DIR/'09_survival_class_embarkation.png',dpi=150,bbox_inches='tight'); plt.close()

    # EDA-stage standardization only; does not feed modeling.
    clean['age_z'] = (clean['age'] - clean['age'].mean()) / clean['age'].std()
    clean['fare_z'] = (clean['fare'] - clean['fare'].mean()) / clean['fare'].std()
    z_summary = clean[['age_z','fare_z']].agg(['mean','std']).round(6)

    # Save tables.
    sex_rates.to_csv(OUT_DIR / 'survival_by_sex.csv', index=False)
    class_rates.to_csv(OUT_DIR / 'survival_by_pclass.csv', index=False)
    sex_class_rates.to_csv(OUT_DIR / 'survival_by_sex_pclass.csv', index=False)
    corr.to_csv(OUT_DIR / 'correlation_matrix.csv')

    report = f'''# EDA Results\n\n## Load and profile\n\nRaw source used: **{source}**. Raw shape: **{raw_shape[0]} rows × {raw_shape[1]} columns**. The connected path calls `sns.load_dataset('titanic')` once and immediately writes `titanic.csv`; the committed CSV is the offline fallback.\n\n### Raw `info()` output\n\n```text\n{raw_info}\n```\n\n### Raw `describe()` output\n\n```text\n{raw_desc}\n```\n\n### Missing values before cleaning\n\n{missing.to_frame('missing_percent').round(2).to_markdown()}\n\n### Cleaning decisions\n\n- `age`: {missing.get('age',0):.2f}% missing, so it falls in the 5%-30% band. Median imputation was used; the median before imputation was **{age_median:.2f}**.\n- `embarked`: {missing.get('embarked',0):.2f}% missing, below 5%, so the affected rows were dropped.\n- `deck`: {missing.get('deck',0):.2f}% missing, above 30%. It was dropped because the missingness is too high for reliable imputation and deck is not required for the main modeling task.\n\nCleaned shape: **{clean.shape[0]} rows × {clean.shape[1]-2} original columns** (plus EDA-only z-score columns).\n\n### Profiling output\n\n```text\n{info_lines}\n```\n\n### Descriptive statistics\n\n```text\n{desc}\n```\n\n## Univariate analysis\n\n- Age IQR bounds: **[{age_lo:.2f}, {age_hi:.2f}]**; outliers: **{age_out}**.\n- Fare IQR bounds: **[{fare_lo:.2f}, {fare_hi:.2f}]**; outliers: **{fare_out}**.\n- Fare mean: **{fare_mean:.2f}**; median: **{fare_median:.2f}**; mode: **{fare_mode:.2f}**. The ordering mean > median > mode indicates a **{skew}** distribution.\n\n## Bivariate survival rates\n\n### By sex\n\n- Female: **{female_rate:.3f} ({female_rate*100:.1f}%)**\n- Male: **{male_rate:.3f} ({male_rate*100:.1f}%)**\n\n### By passenger class\n\n- First: **{first_rate:.3f} ({first_rate*100:.1f}%)**\n- Second: **{second_rate:.3f} ({second_rate*100:.1f}%)**\n- Third: **{third_rate:.3f} ({third_rate*100:.1f}%)**\n\n### Boolean-mask examples for sex + class\n\n- Female & First: **{female_first_rate:.3f} ({female_first_rate*100:.1f}%)**\n- Male & Third: **{male_third_rate:.3f} ({male_third_rate*100:.1f}%)**\n\n### Full sex × class table\n\n{sex_class_rates.to_markdown(index=False)}\n\n## Correlation matrix\n\nThe matrix uses exactly: `{', '.join(corr_cols)}`. The boolean-derived `adult_male` and `alone` columns are excluded as required.\n\nTop two absolute off-diagonal correlations:\n\n1. **{top2[0][0]} ↔ {top2[0][1]}**: r = **{top2[0][2]:.3f}**. This is the strongest linear association in the selected numeric feature set.\n2. **{top2[1][0]} ↔ {top2[1][1]}**: r = **{top2[1][2]:.3f}**. This is the second strongest association by absolute correlation.\n\n## Multivariate data story\n\n**Chart 1 — Survival by sex and class:** Survival differs strongly by sex, and passenger class further separates outcomes. Female passengers generally have much higher survival rates, while third-class passengers have lower survival than first-class passengers.\n\n**Chart 2 — Fare by survival:** Survivors tend to have higher fares than non-survivors, consistent with the relationship between fare and passenger class. The wide spread and high-value fares also explain why fare has substantial IQR outliers.\n\n**Chart 3 — Age vs fare by survival:** The scatter shows survival is not explained by age alone; survival also varies across fare levels. Younger passengers include many survivors, but fare/class-related differences remain visible across age groups.\n\n**Chart 4 — Survival by class and embarkation:** Passenger class is a strong separator of survival, while embarkation adds another grouping dimension. This supports including class and embarkation as predictive features rather than relying on a single demographic variable.\n\n## EDA-stage standardization sanity check\n\n{z_summary.to_markdown()}\n\nThe standardized `age_z` and `fare_z` columns have means approximately 0 and standard deviations approximately 1. These columns are used only for the EDA sanity check and are **not** fed into the modeling pipeline.\n'''
    (OUT_DIR / 'eda_results.md').write_text(report, encoding='utf-8')
    print(report)

if __name__ == '__main__':
    main()
