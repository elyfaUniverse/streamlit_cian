import numpy as np
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

def regression_metrics(y_true, y_pred, y_true_exp=None, y_pred_exp=None):
    """
    Расчёт метрик регрессии.
    Если y_true_exp и y_pred_exp не переданы, вычисляет exp(y_true) и exp(y_pred).
    """
    if y_true_exp is None:
        y_true_exp = np.exp(y_true)
        y_pred_exp = np.exp(y_pred)
    metrics = {
        'MAE (log)': mean_absolute_error(y_true, y_pred),
        'RMSE (log)': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE (price)': mean_absolute_error(y_true_exp, y_pred_exp),
        'RMSE (price)': np.sqrt(mean_squared_error(y_true_exp, y_pred_exp)),
        'MAPE (%)': mean_absolute_percentage_error(y_true_exp, y_pred_exp) * 100,
        'R²': r2_score(y_true, y_pred)
    }
    return metrics

#1.Загрузка модели и бинов 
@st.cache_resource
def load_model_and_bins():
    try:
        model = joblib.load('catboost_final_model.pkl')
        with open('bins.json', 'r') as f:
            bins = json.load(f)
        lat_intervals = bins['lat_intervals']
        lng_intervals = bins['lng_intervals']
        lat_bin_labels = bins['lat_bins']
        lng_bin_labels = bins['lng_bins']
        return model, lat_intervals, lng_intervals, lat_bin_labels, lng_bin_labels
    except FileNotFoundError as e:
        st.error(f"Файл не найден: {e}")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"Ошибка загрузки: {e}")
        return None, None, None, None, None

@st.cache_resource
def load_split_data():
    data = joblib.load('split_data.pkl')
    return data

def show_model_analysis():
    model, lat_intervals, lng_intervals, lat_bin_labels, lng_bin_labels = load_model_and_bins()
    if model is None:
        st.stop()

    features = model.feature_names_
    cat_indices = model.get_cat_feature_indices()
    cat_features = [features[i] for i in cat_indices]

    data = load_split_data()
    X_test_raw = data['X_test'].copy()
    y_test = data['y_test'].copy()
    cv_results = data['cv_results']

    #Предобработка как при обучении
    X_test = X_test_raw.copy()
    X_test['rooms'] = X_test['rooms'].replace('studio', '0')
    X_test['rooms'] = X_test['rooms'].replace('free_plan', '0')
    X_test['rooms'] = X_test['rooms'].astype(int)
    X_test['area_per_room'] = X_test['area_total'] / (X_test['rooms'] + 1)
    feature_names = X_test.columns.tolist()
    y_pred_log = model.predict(X_test)

    # Метрики качества
    metrics = regression_metrics(y_test, y_pred_log)
    metrics = regression_metrics(y_test, y_pred_log)
    st.subheader("Метрики качества на тестовой выборке")

    col1, col2 = st.columns(2)
    col1.metric("MAE (log)", f"{metrics['MAE (log)']:.3f}")
    col2.metric("RMSE (log)", f"{metrics['RMSE (log)']:.3f}")
    col3, col4 = st.columns(2)
    col3.metric("MAE (price)", f"{metrics['MAE (price)']:.3f}")
    col4.metric("RMSE (price)", f"{metrics['RMSE (price)']:.3f}")
    col5, col6 = st.columns(2)
    col5.metric("MAPE (%)", f"{metrics['MAPE (%)']:.3f}")
    col6.metric("R²", f"{metrics['R²']:.3f}")

    y_test_price = np.exp(y_test)
    y_pred_price = np.exp(y_pred_log)

    # График предсказанных vs реальных цен
    st.subheader("Графики")
    df_pred = pd.DataFrame({
        'Реальная цена': y_test_price,
        'Предсказанная цена': y_pred_price
    })
    all_vals = np.concatenate([df_pred['Реальная цена'].values, df_pred['Предсказанная цена'].values])
    min_val = all_vals.min()
    max_val = all_vals.max()

    fig1 = px.scatter(
        df_pred, x='Реальная цена', y='Предсказанная цена',
        log_x=True, log_y=True,
        title='Сравнение предсказанных и реальных цен (лог. шкала)',
        labels={'Реальная цена': 'Реальная цена (руб)', 'Предсказанная цена': 'Предсказанная цена (руб)'},
        range_x=[min_val, max_val], range_y=[min_val, max_val]
    )
    fig1.update_traces(
        marker=dict(color='blue', size=8, line=dict(width=1, color='black')),
        opacity=0.6
    )
    fig1.update_layout(hovermode='closest', height=600, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig1, use_container_width=True)

    #Зависимость от важных признаков
    st.subheader("Зависимость предсказанной цены от важных признаков")
    key_features = ['build_year', 'area_total', 'center_travel_km', 'log_area']
    X_test_for_plot = X_test.copy()
    X_test_for_plot['predicted_price'] = y_pred_price

    col3, col4 = st.columns(2)
    with col3:
        if 'build_year' in X_test_for_plot.columns:
            df_group = X_test_for_plot.groupby('build_year')['predicted_price'].mean().reset_index()
            fig = px.line(df_group, x='build_year', y='predicted_price', markers=True,
                          labels={'build_year': 'Год постройки', 'predicted_price': 'Цена'},
                          title='Зависимость цены от года постройки')
            fig.update_traces(line=dict(color='green', width=1.5), marker=dict(size=10, symbol='circle', line=dict(width=1.5, color='black')), opacity=0.9)
            st.plotly_chart(fig, use_container_width=True)
    with col4:
        if 'area_total' in X_test_for_plot.columns:
            df_group = X_test_for_plot.groupby('area_total')['predicted_price'].mean().reset_index()
            fig = px.line(df_group, x='area_total', y='predicted_price', markers=True,
                          labels={'area_total': 'Общая площадь м²', 'predicted_price': 'Цена'},
                          title='Зависимость цены от общей площади')
            fig.update_traces(line=dict(color='blue', width=1.5), marker=dict(size=10, symbol='circle', line=dict(width=1.5, color='black')), opacity=0.9)
            st.plotly_chart(fig, use_container_width=True)

    col5, col6 = st.columns(2)
    with col5:
        if 'center_travel_km' in X_test_for_plot.columns:
            df_group = X_test_for_plot.groupby('center_travel_km')['predicted_price'].mean().reset_index()
            fig = px.line(df_group, x='center_travel_km', y='predicted_price', markers=True,
                          labels={'center_travel_km': 'Расстояние до центра км', 'predicted_price': 'Цена'},
                          title='Зависимость цены от расстояния до центра')
            fig.update_traces(line=dict(color='violet', width=1.5), marker=dict(size=10, symbol='circle', line=dict(width=1.5, color='black')), opacity=0.9)
            st.plotly_chart(fig, use_container_width=True)
    with col6:
        if 'log_area' in X_test_for_plot.columns:
            df_group = X_test_for_plot.groupby('log_area')['predicted_price'].mean().reset_index()
            fig = px.line(df_group, x='log_area', y='predicted_price', markers=True,
                          labels={'log_area': 'Логарифм площади', 'predicted_price': 'Цена'},
                          title='Зависимость цены от логарифма площади')
            fig.update_traces(line=dict(color='orange', width=1.5), marker=dict(size=10, symbol='circle', line=dict(width=1.5, color='black')), opacity=0.9)
            st.plotly_chart(fig, use_container_width=True)

    #Важность признаков
    st.subheader("Важность признаков (Feature Importance)")
    importances = model.get_feature_importance()
    df_imp = pd.DataFrame({'Признак': feature_names, 'Важность': importances})
    df_imp = df_imp.sort_values('Важность', ascending=False)

    threshold = df_imp['Важность'].median()
    df_imp['Важность_смещ'] = df_imp['Важность'] - threshold
    df_imp['Группа'] = df_imp['Важность_смещ'].apply(
        lambda x: 'Более важные' if x > 0 else ('Менее важные' if x < 0 else 'Медиана')
    )
    df_imp['Важность_округленная'] = df_imp['Важность'].round(4)
    df_imp['Метка'] = df_imp.apply(lambda row: f"{row['Признак']}: {row['Важность_округленная']:.4f}", axis=1)

    color_map = {'Более важные': 'blue', 'Менее важные': '#ff7f0e', 'Медиана': '#d3d3d3'}
    fig_imp = px.bar(df_imp, x='Важность_смещ', y='Признак', orientation='h',
                     title='Вклад признаков относительно медианы (→ важные, ← неважные)',
                     labels={'Важность_смещ': 'Отклонение от медианы', 'Признак': ''},
                     color='Группа', color_discrete_map=color_map, text='Метка')
    fig_imp.update_traces(textposition='outside', cliponaxis=False)
    fig_imp.update_layout(height=600, margin=dict(l=0, r=0, t=40, b=0))
    fig_imp.update_yaxes(showticklabels=False)
    fig_imp.add_vline(x=0, line_dash="dash", line_color="black",
                      annotation_text="← Менее важные | Медиана | Более важные →",
                      annotation_position="top")
    st.plotly_chart(fig_imp, use_container_width=True)

    #Таблица
    st.subheader("Таблица важности признаков")
    df_imp_table = df_imp.sort_values('Важность', ascending=False)
    df_imp_table['Относительная важность'] = df_imp_table['Важность'] / df_imp_table['Важность'].max()
    df_imp_table['Важность (формат)'] = df_imp_table['Важность'].apply(lambda x: f"{x:.3f}")
    st.dataframe(df_imp_table[['Признак', 'Важность', 'Относительная важность']],
                 use_container_width=True, hide_index=True)

    st.subheader("Кросс-валидация 5 фолдов")
    st.dataframe(cv_results.round(4))
    st.write("**Средние метрики по фолдам:**")
    st.dataframe(cv_results.drop(columns='fold').agg(['mean', 'std']).round(4))