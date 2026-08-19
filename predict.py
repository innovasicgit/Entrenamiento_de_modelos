import numpy as np
import pandas as pd
import streamlit as st

from .model_utils import pred_threshold


def _is_normal_like_label(label):
    text = str(label).strip().lower()
    normal_tokens = {
        "0",
        "normal",
        "benigno",
        "benign",
        "legitimo",
        "clean",
        "safe",
    }
    return text in normal_tokens


def _infer_attack_class_index(classes):
    if classes is None or len(classes) == 0:
        return 1

    classes_arr = np.asarray(classes)

    idx_ones = np.where(classes_arr == 1)[0]
    if idx_ones.size > 0:
        return int(idx_ones[0])

    non_normal = [i for i, c in enumerate(classes_arr) if not _is_normal_like_label(c)]
    if len(non_normal) == 1:
        return int(non_normal[0])

    if len(classes_arr) >= 2:
        return 1

    return 0


def _sanitize_numeric_frame(data):
    clean = data.copy()
    clean = clean.replace([np.inf, -np.inf], np.nan)
    clean = clean.apply(pd.to_numeric, errors="coerce")
    clean = clean.fillna(0)
    return clean


def predecir(model, data, model_option):
    try:
        data = data.copy()

        # 1) Resolver nombres esperados por el modelo
        expected = None
        if hasattr(model, "feature_names_in_"):
            expected = list(model.feature_names_in_)
        elif hasattr(model, "named_steps"):
            prepro = (
                model.named_steps.get("prepro_2_del")
                or model.named_steps.get("preprocessor")
                or model.named_steps.get("preprocessor_3_pca")
            )
            if prepro is not None and hasattr(prepro, "get_feature_names_out"):
                try:
                    expected = list(prepro.get_feature_names_out())
                except Exception:
                    expected = None

        if expected is not None:
            for col in expected:
                if col not in data.columns:
                    if "__" in col:
                        raw_name = col.split("__", 1)[1]
                        if raw_name in data.columns:
                            data[col] = pd.to_numeric(data[raw_name], errors="coerce").fillna(0)
                        else:
                            data[col] = 0
                    else:
                        data[col] = 0

            data = data[expected]

        # 2) Sanitizar tipos
        if model_option in ["IForest", "OCSVM", "KMEANS", "AUTOENCODER"]:
            for col in data.columns:
                if data[col].dtype == object:
                    lowered = data[col].astype(str).str.strip().str.lower()
                    data[col] = lowered.replace({
                        "": np.nan,
                        "nan": np.nan,
                        "none": np.nan,
                        "null": np.nan,
                    })
            data = _sanitize_numeric_frame(data)

        columnas_pca2 = [
            "componente1", "componente2", "componente3",
            "componente4", "componente5", "componente6"
        ]

        scores = None
        predicciones = None

        if model_option == "IForest":
            scores = model.decision_function(data)
            pred_raw = model.predict(data)
            predicciones = np.where(pred_raw == 1, 1, 0)

        elif model_option == "OCSVM":
            try:
                scores = model.decision_function(data)
            except Exception:
                scores = None

            if st.session_state.get("ocsvm_use_sign", True):
                pred_raw = model.predict(data)
                predicciones = np.where(pred_raw == 1, 1, 0)
            else:
                if scores is None:
                    predicciones = model.predict(data)
                    predicciones = np.where(predicciones == 1, 1, 0)
                else:
                    pct = float(st.session_state.get("ocsvm_percentile", 0.45))
                    threshold = np.percentile(scores, 100 * pct)
                    predicciones = np.where(scores >= threshold, 1, 0)

        elif model_option == "KMEANS":
            # Debug opcional
            # print("Esperadas:", list(model.feature_names_in_))
            # print("Entrantes:", list(data.columns))
            expected = list(model.feature_names_in_)

            for col in expected:
                if col not in data.columns:
                    if "__" in col:
                        raw_name = col.split("__", 1)[1]
                        if raw_name in data.columns:
                            data[col] = pd.to_numeric(data[raw_name], errors="coerce").fillna(0)
                        else:
                            data[col] = 0
                    else:
                        data[col] = 0

            data = data[expected]
            
            scores = None
            predicciones = model.predict(data)

        elif model_option == "AUTOENCODER":
            proba = None
            est = None
            try:
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(data)
                est = getattr(model, "named_steps", {}).get("autoencoder_classifier", None)
                if proba is None and est is not None and hasattr(est, "predict_proba"):
                    proba = est.predict_proba(data)
            except Exception:
                proba = None

            rec_err = None
            try:
                if hasattr(model, "reconstruction_error"):
                    rec_err = model.reconstruction_error(data)
                elif est is not None and hasattr(est, "reconstruction_error"):
                    rec_err = est.reconstruction_error(data)
            except Exception:
                rec_err = None

            normal_idx = 0
            thr = st.session_state.get("ae_normal_threshold", 0.60)
            rec_thr = st.session_state.get("ae_reconstruction_threshold", 0.02)

            if isinstance(proba, np.ndarray):
                pred_list = []
                for i, row in enumerate(proba):
                    is_normal_proba = row[normal_idx] >= thr
                    is_normal_recon = (rec_err is not None and i < len(rec_err) and rec_err[i] <= rec_thr)
                    if is_normal_proba and (rec_err is None or is_normal_recon):
                        pred_list.append(0)
                    else:
                        pred_list.append(int(np.argmax(row)))
                predicciones = np.array(pred_list, dtype=int)
                scores = row[normal_idx] if "row" in locals() else proba[:, normal_idx]
            else:
                try:
                    predicciones = model.predict(data)
                except Exception:
                    predicciones = est.predict(data) if est is not None else np.zeros(len(data), dtype=int)
                scores = None

        else:
            try:
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(data)
                    classes = getattr(model, "classes_", None)
                    if isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] >= 2:
                        attack_idx = _infer_attack_class_index(classes)
                        attack_idx = max(0, min(attack_idx, proba.shape[1] - 1))
                        scores = proba[:, attack_idx]

                        attack_threshold = float(st.session_state.get("supervised_attack_threshold", 0.70))

                        if classes is not None and len(classes) == proba.shape[1]:
                            classes_arr = np.asarray(classes)
                            if proba.shape[1] == 2:
                                normal_idx = 0 if attack_idx == 1 else 1
                                pred_idx = np.where(scores >= attack_threshold, attack_idx, normal_idx)
                                predicciones = classes_arr[pred_idx]
                            else:
                                pred_idx = np.argmax(proba, axis=1)
                                pred_idx = np.where(scores >= attack_threshold, attack_idx, pred_idx)
                                predicciones = classes_arr[pred_idx]
                        else:
                            if proba.shape[1] == 2:
                                normal_idx = 0 if attack_idx == 1 else 1
                                predicciones = np.where(scores >= attack_threshold, attack_idx, normal_idx)
                            else:
                                predicciones = np.argmax(proba, axis=1)

                    if predicciones is None:
                        predicciones = model.predict(data)
            except Exception as pred_err:
                expected_n = getattr(model, "n_features_in_", None)
                detail = ""
                if expected_n is not None:
                    detail = f" El modelo espera {expected_n} features y recibió {data.shape[1]}."
                raise RuntimeError(
                    f"Fallo en predict del modelo supervisado '{model_option}'."
                    f" Verifica columnas/orden de features de entrada.{detail}"
                ) from pred_err

        if predicciones is None:
            predicciones = np.zeros(len(data), dtype=int)

        predicciones = np.asarray(predicciones).reshape(-1)

        pp3 = None
        try:
            if model_option in ["IForest", "OCSVM"]:
                transformer = None
                if hasattr(model, "named_steps"):
                    transformer = model.named_steps.get("preprocessor_3_pca") or model.named_steps.get("prepro_3_pca")
                if transformer is None and hasattr(model, "__getitem__"):
                    transformer = model[0]
                if transformer is not None and hasattr(transformer, "transform"):
                    pp3_arr = transformer.transform(data)
                    if isinstance(pp3_arr, np.ndarray) and pp3_arr.shape[1] == 6:
                        pp3 = pd.DataFrame(pp3_arr, columns=columnas_pca2)
        except Exception:
            pp3 = None

        if pp3 is None:
            pp3 = data.select_dtypes(include=[np.number]).copy()

        pp3["cluster"] = predicciones
        return pp3, predicciones, scores

    except Exception as e:
        st.error(f"Error en la predicción: {str(e)}")
        import traceback
        st.error(f"Detalles: {traceback.format_exc()}")
        return None, None, None


def mostrar_metricas(silhouette, calinski, davies):
    st.markdown("### 📊 Métricas Internas")
    cols = st.columns(3)

    with cols[0]:
        st.metric(
            "Silhouette Score",
            f"{silhouette:.3f}",
            delta="Bueno" if silhouette > 0.5 else "Regular",
        )

    with cols[1]:
        st.metric(
            "Calinski Score",
            f"{calinski:.3f}",
            delta="Bueno" if calinski > 1000 else "Regular",
        )

    with cols[2]:
        st.metric(
            "Davies Score",
            f"{davies:.3f}",
            delta="Bueno" if davies < 0.5 else "Regular",
        )