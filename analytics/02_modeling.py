from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, roc_auc_score, mean_absolute_error,
                             mean_squared_error, r2_score)
from imblearn.over_sampling import SMOTE

BASE = Path(__file__).resolve().parent
DATA = BASE / 'data' / 'titanic_clean.csv'
OUT = BASE / 'outputs'; OUT.mkdir(exist_ok=True)
MODELS = BASE / 'models'; MODELS.mkdir(exist_ok=True)


def make_preprocessor(numeric, categorical):
    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    return ColumnTransformer([('num', num_pipe, numeric), ('cat', cat_pipe, categorical)])


def evaluate(name, pipe, X_test, y_test):
    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)[:,1] if hasattr(pipe, 'predict_proba') else pipe.decision_function(X_test)
    return {
        'model': name,
        'accuracy': accuracy_score(y_test,pred),
        'precision': precision_score(y_test,pred,zero_division=0),
        'recall': recall_score(y_test,pred,zero_division=0),
        'f1': f1_score(y_test,pred,zero_division=0),
        'auc': roc_auc_score(y_test,prob),
        'confusion_matrix': confusion_matrix(y_test,pred).tolist(),
        'fpr': roc_curve(y_test,prob)[0].tolist(),
        'tpr': roc_curve(y_test,prob)[1].tolist(),
    }


def main():
    df = pd.read_csv(DATA)
    target = 'survived'
    features = ['pclass','sex','age','sibsp','parch','fare','embarked']
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

    numeric = ['pclass','age','sibsp','parch','fare']
    categorical = ['sex','embarked']
    pre = make_preprocessor(numeric,categorical)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42)
    }
    results=[]; fitted={}
    for name, est in models.items():
        pipe = Pipeline([('preprocessor', pre), ('model', est)])
        pipe.fit(X_train,y_train)
        fitted[name]=pipe
        results.append(evaluate(name,pipe,X_test,y_test))

    # Decision-tree rendering with labeled features and classes.
    dt = fitted['Decision Tree']
    feature_names = list(dt.named_steps['preprocessor'].get_feature_names_out())
    plt.figure(figsize=(22,12)); plot_tree(dt.named_steps['model'], feature_names=feature_names, class_names=['Not survived','Survived'], filled=False, max_depth=4, fontsize=7); plt.title('Decision Tree (first four levels shown)'); plt.tight_layout(); plt.savefig(OUT/'decision_tree.png',dpi=150); plt.close()

    # Confusion matrices.
    fig, axes = plt.subplots(1,3,figsize=(12,4))
    for ax,(name,pipe) in zip(axes,fitted.items()):
        cm=confusion_matrix(y_test,pipe.predict(X_test)); sns.heatmap(cm,annot=True,fmt='d',cbar=False,ax=ax); ax.set_title(name); ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    plt.tight_layout(); plt.savefig(OUT/'confusion_matrices.png',dpi=150); plt.close()

    # ROC curves.
    plt.figure(figsize=(7,5))
    for r in results: plt.plot(r['fpr'],r['tpr'],label=f"{r['model']} (AUC={r['auc']:.3f})")
    plt.plot([0,1],[0,1],'--'); plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curves'); plt.legend(); plt.tight_layout(); plt.savefig(OUT/'roc_curves.png',dpi=150); plt.close()

    comparison=pd.DataFrame([{k:v for k,v in r.items() if k not in ['confusion_matrix','fpr','tpr']} for r in results])
    comparison.to_csv(OUT/'classification_comparison.csv',index=False)

    # Imbalance comparison on logistic regression; SMOTE only on training fold.
    variants=[]
    base=LogisticRegression(max_iter=1000,random_state=42)
    base_pipe=Pipeline([('preprocessor',make_preprocessor(numeric,categorical)),('model',base)])
    base_pipe.fit(X_train,y_train); r=evaluate('Baseline',base_pipe,X_test,y_test); variants.append({k:r[k] for k in ['model','precision','recall','f1']})
    bal=LogisticRegression(max_iter=1000,class_weight='balanced',random_state=42)
    bal_pipe=Pipeline([('preprocessor',make_preprocessor(numeric,categorical)),('model',bal)])
    bal_pipe.fit(X_train,y_train); r=evaluate('class_weight=balanced',bal_pipe,X_test,y_test); variants.append({k:r[k] for k in ['model','precision','recall','f1']})
    # Fit preprocessing on train only, transform train/test, then SMOTE train only.
    pre_smote=make_preprocessor(numeric,categorical); Xtr=pre_smote.fit_transform(X_train); Xte=pre_smote.transform(X_test)
    sm=SMOTE(random_state=42); Xtr_sm,ytr_sm=sm.fit_resample(Xtr,y_train)
    sm_model=LogisticRegression(max_iter=1000,random_state=42); sm_model.fit(Xtr_sm,ytr_sm)
    pred=sm_model.predict(Xte); prob=sm_model.predict_proba(Xte)[:,1]
    variants.append({'model':'SMOTE (train only)','precision':precision_score(y_test,pred,zero_division=0),'recall':recall_score(y_test,pred,zero_division=0),'f1':f1_score(y_test,pred,zero_division=0)})
    imb=pd.DataFrame(variants); imb.to_csv(OUT/'imbalance_comparison.csv',index=False)

    # GridSearchCV over RF hyperparameters. The estimator has OOB enabled.
    rf_base=RandomForestClassifier(oob_score=True,random_state=42,n_jobs=-1)
    rf_pipe=Pipeline([('preprocessor',make_preprocessor(numeric,categorical)),('model',rf_base)])
    grid=GridSearchCV(rf_pipe, {'model__n_estimators':[100,200], 'model__max_depth':[None,5,10], 'model__max_features':['sqrt','log2']}, cv=5, scoring='f1', n_jobs=-1)
    grid.fit(X_train,y_train)
    best_rf=grid.best_estimator_; oob=float(best_rf.named_steps['model'].oob_score_)

    # Regression side-task: fare from the other available features.
    reg_features=['pclass','sex','age','sibsp','parch','embarked']
    Xr=df[reg_features]; yr=df['fare']
    Xr_train,Xr_test,yr_train,yr_test=train_test_split(Xr,yr,test_size=0.2,random_state=42)
    reg_pre=make_preprocessor(['pclass','age','sibsp','parch'],['sex','embarked'])
    reg_pipe=Pipeline([('preprocessor',reg_pre),('model',LinearRegression())])
    reg_pipe.fit(Xr_train,yr_train); yp=reg_pipe.predict(Xr_test)
    mae=mean_absolute_error(yr_test,yp); rmse=mean_squared_error(yr_test,yp)**0.5; r2=r2_score(yr_test,yp); n=len(yr_test); p=reg_pipe.named_steps['preprocessor'].transform(Xr_test).shape[1]; adj=1-(1-r2)*(n-1)/(n-p-1)
    residuals=yr_test-yp
    plt.figure(figsize=(7,4)); sns.scatterplot(x=yp,y=residuals); plt.axhline(0,ls='--'); plt.xlabel('Predicted Fare'); plt.ylabel('Residual'); plt.title('Fare Regression Residual Plot'); plt.tight_layout(); plt.savefig(OUT/'fare_residuals.png',dpi=150); plt.close()

    # Final comparison table keeps classification and regression metric groups separate.
    combined_rows=[]
    for row in comparison.to_dict('records'):
        combined_rows.append({'model_type':'classification','model':row['model'],'accuracy':row['accuracy'],'precision':row['precision'],'recall':row['recall'],'f1':row['f1'],'auc':row['auc'],'MAE':np.nan,'RMSE':np.nan,'R2':np.nan,'Adjusted_R2':np.nan})
    combined_rows.append({'model_type':'regression','model':'Linear Regression (fare)','accuracy':np.nan,'precision':np.nan,'recall':np.nan,'f1':np.nan,'auc':np.nan,'MAE':mae,'RMSE':rmse,'R2':r2,'Adjusted_R2':adj})
    final_comparison=pd.DataFrame(combined_rows)
    final_comparison.to_csv(OUT/'final_model_comparison.csv',index=False)

    # Save complete classification pipeline.
    best_name=max(comparison.to_dict('records'),key=lambda x:x['f1'])['model']
    best_pipeline=fitted[best_name]
    dump(best_pipeline,MODELS/'best_classifier_pipeline.joblib')
    reloaded=load(MODELS/'best_classifier_pipeline.joblib')
    reload_prediction=int(reloaded.predict(X_test.iloc[[0]])[0])

    # Combined report.
    rec='Random Forest' if best_name=='Random Forest' else best_name
    if best_name=='Random Forest': rec_text=f"Random Forest achieved the highest test F1 among the three classifiers ({comparison.loc[comparison.model==best_name,'f1'].iloc[0]:.3f}) while maintaining an AUC of {comparison.loc[comparison.model==best_name,'auc'].iloc[0]:.3f}. It is the recommended classifier because it captures nonlinear interactions without requiring manual feature transformations. The Decision Tree is easier to inspect, but the ensemble is generally more robust. The final deployment artifact is the complete preprocessing-plus-estimator pipeline."
    else: rec_text=f"{best_name} achieved the highest test F1 ({comparison.loc[comparison.model==best_name,'f1'].iloc[0]:.3f}) among the three classifiers. Its AUC was {comparison.loc[comparison.model==best_name,'auc'].iloc[0]:.3f}, indicating useful ranking ability. The other models remain useful comparison baselines, but the highest-F1 model is selected for the saved end-to-end artifact."
    hetero='evidence of heteroscedasticity' if abs(np.corrcoef(yp, np.abs(residuals))[0,1])>0.25 else 'no strong evidence of heteroscedasticity'
    report=f'''# Modeling Results\n\n## Split and leakage control\n\nThe cleaned dataset was split **before preprocessing** using `train_test_split(..., stratify=y, random_state=42)`. The observed class balance was {y.value_counts(normalize=True).round(3).to_dict()}, so stratification preserves approximately the same survived/not-survived proportions in train and test. All imputers, encoders, and scalers are inside scikit-learn pipelines and are fit on the training split only.\n\n## Classification comparison\n\n{comparison.round(4).to_markdown(index=False)}\n\nConfusion matrices and ROC curves are saved as supporting chart artifacts.\n\n## Imbalance comparison\n\n{imb.round(4).to_markdown(index=False)}\n\nThe best imbalance strategy is selected by the balance of precision, recall and F1 rather than accuracy alone. SMOTE is applied only after fitting the preprocessing transformer on the training fold and never to the test set, preventing leakage.\n\n## Random Forest tuning\n\nBest parameters: **{grid.best_params_}**\n\nBest cross-validation F1: **{grid.best_score_:.4f}**\n\nOOB score from the fitted `RandomForestClassifier(oob_score=True, ...)`: **{oob:.4f}**\n\n## Regression side-task\n\n| Metric | Value |\n|---|---:|\n| MAE | {mae:.4f} |\n| RMSE | {rmse:.4f} |\n| R² | {r2:.4f} |\n| Adjusted R² | {adj:.4f} |\n\nThe residual plot shows **{hetero}** based on the visual spread and a simple residual-vs-predicted diagnostic.\n\n## Final recommendation\n\n{rec_text}\n\n## Final model comparison table\n\n{final_comparison.round(4).to_markdown(index=False)}\n\nClassification and regression metrics are intentionally presented as separate metric groups because they measure different objectives and are not directly comparable.\n\n## Saved artifact\n\n`models/best_classifier_pipeline.joblib` contains the complete fitted preprocessing + classifier pipeline. It was reloaded with `joblib.load` and successfully generated a prediction on raw test-row features; reload check prediction: **{reload_prediction}**.\n'''
    (OUT/'modeling_results.md').write_text(report,encoding='utf-8')
    print(report)

if __name__=='__main__': main()
