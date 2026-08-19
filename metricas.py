#Proyecto IDS Tesis Fernando Gutierrez P.
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import info_sistema


LABEL_COLUMN_NAME_HINTS = {
    "label",
    "labels",
    "class",
    "target",
    "y",
    "is_attack",
    "attack",
    "tipo_ataque",
    "intrusion",
    "anomaly",
    "categoria",
}


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def _is_normal_label(value):
    """Heurística para detectar etiquetas de tráfico normal."""
    if pd.isna(value):
        return True

    if isinstance(value, (int, np.integer, float, np.floating)):
        return int(value) == 0

    text = str(value).strip().lower()
    normal_tokens = {
        "0",
        "normal",
        "benigno",
        "benign",
        "legitimo",
        "legítimo",
        "clean",
        "safe",
    }
    return text in normal_tokens


def to_binary_attack_labels(y):
    """Convierte etiquetas arbitrarias a binario: 0=Normal, 1=Ataque."""
    arr = np.asarray(y)
    return np.array([0 if _is_normal_label(v) else 1 for v in arr], dtype=int)


def is_plausible_label_series(series, max_unique=20):
    """Valida si una serie parece una columna de etiquetas y no una feature continua."""
    if series is None:
        return False

    s = pd.Series(series)
    s = s.dropna()
    if s.empty:
        return False

    # Evita columnas continuas tipo timestamp/delta_time.
    unique_count = int(s.nunique())
    if unique_count > max_unique:
        return False

    # Acepta binario clásico.
    if pd.api.types.is_numeric_dtype(s):
        uniq = set(pd.Series(s).astype(float).unique().tolist())
        if uniq.issubset({-1.0, 0.0, 1.0}):
            return True

    # Acepta etiquetas de texto con tokens de tráfico normal/ataque.
    lowered = s.astype(str).str.strip().str.lower()
    normal_hits = lowered.map(_is_normal_label).sum()
    attack_tokens = {"attack", "ataque", "intrusion", "intrusión", "malicious", "anomalo", "anómalo", "ddos", "dos"}
    attack_hits = lowered.apply(lambda v: any(tok in v for tok in attack_tokens)).sum()
    return bool(normal_hits > 0 or attack_hits > 0)


def candidate_label_columns(df, max_unique=20):
    """Devuelve columnas candidatas para etiquetas reales en métricas externas."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []

    cols = list(df.columns)
    by_name = [
        c for c in cols
        if any(hint in str(c).strip().lower() for hint in LABEL_COLUMN_NAME_HINTS)
    ]

    valid_by_name = [c for c in by_name if is_plausible_label_series(df[c], max_unique=max_unique)]
    if valid_by_name:
        return valid_by_name

    # Fallback: buscar por cardinalidad/forma si no hay nombres sugerentes.
    return [c for c in cols if is_plausible_label_series(df[c], max_unique=max_unique)]


def metricas_externas(y_true, y_pred, y_score=None):
    """Calcula métricas externas binarias para detección de ataques.

    Devuelve un dict con:
    - Accuracy
    - Precision
    - Recall
    - F1
    - ROC_AUC (si es calculable)
    - MatrizConfusion (2x2 con labels [0, 1])
    """
    y_true_bin = to_binary_attack_labels(y_true)
    y_pred_bin = to_binary_attack_labels(y_pred)

    if y_true_bin.shape[0] != y_pred_bin.shape[0]:
        raise ValueError("y_true e y_pred deben tener la misma longitud")

    out = {
        "Accuracy": round(float(accuracy_score(y_true_bin, y_pred_bin)), 4),
        "Precision": round(float(precision_score(y_true_bin, y_pred_bin, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_true_bin, y_pred_bin, zero_division=0)), 4),
        "F1": round(float(f1_score(y_true_bin, y_pred_bin, zero_division=0)), 4),
        "MatrizConfusion": confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]),
    }

    # ROC AUC requiere ambas clases presentes en y_true.
    if np.unique(y_true_bin).size == 2:
        try:
            if y_score is not None:
                out["ROC_AUC"] = round(float(roc_auc_score(y_true_bin, np.asarray(y_score))), 4)
            elif np.unique(y_pred_bin).size == 2:
                out["ROC_AUC"] = round(float(roc_auc_score(y_true_bin, y_pred_bin)), 4)
        except Exception:
            pass

    return out


def metricas(df,tipo_ataque, Trans_cluster):
    '''
    label: 
    cluster: 
    '''
    print(metrics.classification_report( df[tipo_ataque], df[Trans_cluster] ))
    print('Purity ', round(purity_score(df[tipo_ataque], df[Trans_cluster]),5))
    print('homogeneity_score: ', round(metrics.homogeneity_score(df[tipo_ataque], df[Trans_cluster]),5))
    print('completeness_score: ', round(metrics.completeness_score(df[tipo_ataque], df[Trans_cluster]),5))
    print('v_measure_score: ', round(metrics.v_measure_score(df[tipo_ataque], df[Trans_cluster]),5))
    #print('adjusted_rand_score: ', round(metrics.adjusted_rand_score(y['tipo_ataque_num'], result['predictions']),5))
    print('adjusted_mutual_info_score: ', round(metrics.adjusted_mutual_info_score(df[tipo_ataque], df[Trans_cluster]),5))
    

def metrica_internas(pp3,cluster):
    """Calcula métricas internas de clustering usando scikit-learn.

    Nota para documentación/explicación:
    - Estas métricas se calculan mediante `sklearn.metrics` (no se implementan a mano).
    - Entradas:
        - `pp3`: matriz de características X (n_muestras, n_features)
        - `cluster`: etiquetas de cluster asignadas (y_pred)
    - Funciones usadas (scikit-learn):
        - `metrics.silhouette_score(X, labels, metric=...)` (más alto = mejor)
        - `metrics.calinski_harabasz_score(X, labels)` (más alto = mejor)
        - `metrics.davies_bouldin_score(X, labels)` (más bajo = mejor)
    """

    ss = round(metrics.silhouette_score(pp3, cluster, metric='sqeuclidean'),5)
#     ss='prueba_DBSCAN'
    chs = round(metrics.calinski_harabasz_score(pp3, cluster),5)
    dbs = round(metrics.davies_bouldin_score(pp3, cluster),5)
    return (ss,chs,dbs)
#     return (chs,dbs)

    


#Asignar los nombres a los clusters predichos acorde con las categorias conocidas(tipo_ataque)=(y) 
def y(df,start_, n_clusters,cluster,tipo_ataque):
    l=[]
    for ClusterNum in range(start_, n_clusters):

        OneCluster = pd.DataFrame(df[df[cluster] == ClusterNum].groupby(tipo_ataque).size())
        OneCluster.columns=['Size']
    #     print(f'{OneCluster}')
        NewDigit = OneCluster.index[OneCluster['Size'] == OneCluster['Size'].max()].tolist()
        NewDigit[0]
    #     print(f'{NewDigit[0]}')

        rowIndex = df.index[df[cluster] == ClusterNum]
        df.loc[rowIndex, 'Trans_cluster'] = NewDigit[0]

        print(ClusterNum, NewDigit[0])
        l=l+[(ClusterNum, NewDigit[0])]
    return l

# def specifity(df1):
def matriz(df1,tipo_ataque,Trans_cluster):
    
    cm = metrics.confusion_matrix(df1[tipo_ataque], df1[Trans_cluster], labels=df1[tipo_ataque].value_counts().index)
    cm = pd.DataFrame(cm, columns=df1[tipo_ataque].value_counts().index, index=df1[tipo_ataque].value_counts().index)
    tabla_conf = pd.DataFrame(index=df1[tipo_ataque].value_counts().index, columns=['tp','fp','fn','tn'])
#     display matriz confusion
    matriz = pd.DataFrame(index=['Positive','Negative'], columns=['Positive','Negative'])
    for i in range(cm.shape[0]):

        tp=cm.iloc[i,i]
        fp=cm.iloc[:,i].values.sum()-tp
        fn=cm.iloc[i,:].values.sum()-tp
        tn=cm.iloc[:i,:i].values.sum()+cm.iloc[i+1:,:i].values.sum() +cm.iloc[:i,i+1:].values.sum() +cm.iloc[i+1:,i+1:].values.sum()
        matriz.iloc[0,0]=tp
        matriz.iloc[0,1]=fn
        matriz.iloc[1,0]=fp
        matriz.iloc[1,1]=tn
        print('')
        print(df1[tipo_ataque].value_counts().index[i])
        print(matriz)
        tabla_conf.iloc[i,:] = tp,fp,fn,tn

    tabla_conf['tp+fn'] = tabla_conf['tp']+tabla_conf['fn']
    tabla_conf['tp+fp'] = tabla_conf['tp']+tabla_conf['fp']
    tabla_conf['fn+tn'] = tabla_conf['fn']+tabla_conf['tn']
    tabla_conf['fp+tn'] = tabla_conf['fp']+tabla_conf['tn']
#     recall= tp / (tp + fn)
    tabla_conf['recall'] = tabla_conf['tp']/tabla_conf['tp+fn']
#     precision_score= tp / (tp + fp)
    tabla_conf['precision'] = tabla_conf['tp']/tabla_conf['tp+fp']
#     F1 = 2 * (precision * recall) / (precision + recall)
    tabla_conf['F1'] = (2*tabla_conf['precision']*tabla_conf['recall'])/(tabla_conf['precision']+tabla_conf['recall'])
    
    
    tabla_binaria = pd.DataFrame(index=[0,1], columns=['Cantidad','coincidencia','% prediccion','total prediciones'])
    tabla_binaria.iloc[0,0]= tabla_conf.iloc[0,4]
    tabla_binaria.iloc[1,0]=tabla_conf.iloc[0,7]
    tabla_binaria.iloc[0,1]= tabla_conf.iloc[0,0]
    tabla_binaria.iloc[1,1]=tabla_conf.iloc[0,3]
    tabla_binaria.iloc[0,2]=tabla_binaria.iloc[0,1]/tabla_binaria.iloc[0,0]
    tabla_binaria.iloc[1,2]=tabla_binaria.iloc[1,1]/tabla_binaria.iloc[1,0]
    tabla_binaria.iloc[0,3]=tabla_conf.iloc[0,5]
    tabla_binaria.iloc[1,3]=tabla_conf.iloc[0,6]
    return tabla_binaria ,tabla_conf

