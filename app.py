import re
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Diccionario IATA → nombre completo
IATA_NOMBRES = {
    'LP': 'LATAM Perú',
    'LA': 'LATAM Airlines',
    'UX': 'Air Europa',
    'IB': 'Iberia',
    'AF': 'Air France',
    'EA': 'EA Airways',
    'CM': 'Copa Airlines',
    '2I': 'Star Perú',
    'KL': 'KLM',
    'AR': 'Aerolíneas Argentinas',
    'AC': 'Air Canada',
    'UA': 'United Airlines',
    'MD': 'MD',
    'PU': 'PU',
    'UT': 'UT',
}

# Aeropuertos peruanos para clasificación nacional/internacional
AEROPUERTOS_PERU = {
    'LIM', 'AQP', 'CUZ', 'CUS', 'TRU', 'TPP', 'CJA', 'CIX',
    'IQT', 'PIU', 'PEM', 'AYP', 'HUU', 'JAU', 'JUL', 'PCL', 'TBP', 'TCQ', 'ATA',
}


def clasificar_ruta(ruta) -> str:
    if pd.isna(ruta):
        return 'Sin clasificar'
    codigos = {c for c in re.split(r'[/\-]', str(ruta).upper()) if len(c) == 3 and c.isalpha() and c != 'XXX'}
    if not codigos:
        return 'Sin clasificar'
    return 'Nacional' if codigos.issubset(AEROPUERTOS_PERU) else 'Internacional'


# 1. Configuración de la página
st.set_page_config(page_title="Dashboard Agencia de Viajes", page_icon="✈️", layout="wide")


# 2. Carga y limpieza de datos con caché
@st.cache_data(ttl=3600)
def cargar_datos():
    try:
        df = pd.read_csv('ventas_viajes.csv')

        # Conversión de fechas
        for col in ['Fecha de pedido', 'Fecha de ida', 'Fecha de retorno', 'Fecha de pago']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Conversión numérica
        for col in ['Fee institucional', 'Comisión proveedor', 'Precio total pagado', 'PAGO AL PROVEEDOR']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Comisión total
        if 'Fee institucional' in df.columns and 'Comisión proveedor' in df.columns:
            df['Comisión Total'] = df['Fee institucional'] + df['Comisión proveedor']

        # Margen real basado en costo al proveedor
        if 'PAGO AL PROVEEDOR' in df.columns and 'Precio total pagado' in df.columns:
            df['Margen Real'] = df['Precio total pagado'] - df['PAGO AL PROVEEDOR']
            df['Margen Real %'] = (df['Margen Real'] / df['Precio total pagado']).clip(0, 1) * 100

        # Columnas temporales para gráficos YoY
        if 'Fecha de pedido' in df.columns:
            df['Mes Num'] = df['Fecha de pedido'].dt.month
            df['Mes Nombre'] = df['Fecha de pedido'].dt.strftime('%b')
            df['Mes'] = df['Fecha de pedido'].dt.to_period('M').astype(str)

        # Nombre legible de aerolínea
        if 'Aerolíneas' in df.columns:
            df['Aerolínea Nombre'] = df['Aerolíneas'].map(IATA_NOMBRES).fillna(df['Aerolíneas'] + ' (desconocido)')

        # Clasificación nacional / internacional
        if 'Rutas' in df.columns:
            df['Tipo de Vuelo'] = df['Rutas'].apply(clasificar_ruta)

        return df
    except FileNotFoundError:
        st.error("Error: No se encontró el archivo 'ventas_viajes.csv'. Ejecuta primero 'uv run python main.py'.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error al procesar los datos: {e}")
        return pd.DataFrame()


df = cargar_datos()

if df.empty:
    st.stop()

# 3. Sidebar — Filtros Interactivos
st.sidebar.header("Filtros Interactivos 🔍")

# Filtro por Año
años_disponibles = sorted(df['Año'].dropna().unique().tolist()) if 'Año' in df.columns else []
año_seleccion = st.sidebar.multiselect("Año", años_disponibles, default=años_disponibles)

# Filtro por Estado de Pago
estado_seleccion = st.sidebar.multiselect(
    "Estado de Pago", ["Pagado", "Pendiente"], default=["Pagado", "Pendiente"]
)

# Filtro de Fechas
fechas_validas = df['Fecha de pedido'].dropna() if 'Fecha de pedido' in df.columns else pd.Series(dtype="datetime64[ns]")
if not fechas_validas.empty:
    min_date = fechas_validas.min().date()
    max_date = fechas_validas.max().date()
    fechas_seleccionadas = st.sidebar.date_input(
        "Rango de Fechas de Pedido", [min_date, max_date], min_value=min_date, max_value=max_date
    )
else:
    st.sidebar.warning("No hay fechas válidas en 'Fecha de pedido'.")
    fechas_seleccionadas = []

# Filtro de Aerolíneas
aerolineas = df['Aerolínea Nombre'].dropna().unique().tolist() if 'Aerolínea Nombre' in df.columns else []
aero_seleccion = st.sidebar.multiselect("Aerolíneas", aerolineas, default=aerolineas)

# Aplicar filtros
mask = pd.Series(True, index=df.index)

if año_seleccion and 'Año' in df.columns:
    mask &= df['Año'].isin(año_seleccion)

if estado_seleccion and 'Estado de Pago' in df.columns:
    mask &= df['Estado de Pago'].isin(estado_seleccion)

if len(fechas_seleccionadas) == 2 and 'Fecha de pedido' in df.columns:
    mask &= (
        df['Fecha de pedido'].notna()
        & (df['Fecha de pedido'].dt.date >= fechas_seleccionadas[0])
        & (df['Fecha de pedido'].dt.date <= fechas_seleccionadas[1])
    )

if aero_seleccion and 'Aerolínea Nombre' in df.columns:
    mask &= df['Aerolínea Nombre'].isin(aero_seleccion)

df_filtrado = df[mask]

if df_filtrado.empty:
    st.warning("No hay datos para los filtros seleccionados.")
    st.stop()

# 4. Título + KPIs
st.title("✈️ Dashboard Comercial y Financiero")
st.markdown("---")

col1, col2, col3, col4, col5 = st.columns(5)

ventas_totales = df_filtrado['Precio total pagado'].sum()
comision_total = df_filtrado['Comisión Total'].sum() if 'Comisión Total' in df_filtrado.columns else 0
margen_prom = df_filtrado['Margen Real %'].mean() if 'Margen Real %' in df_filtrado.columns else float('nan')
ticket_promedio = df_filtrado['Precio total pagado'].mean()
monto_pendiente = df_filtrado[df_filtrado['Estado de Pago'] == 'Pendiente']['Precio total pagado'].sum() if 'Estado de Pago' in df_filtrado.columns else 0

col1.metric("Ventas Totales", f"${ventas_totales:,.2f}")
col2.metric("Comisión Total", f"${comision_total:,.2f}")
col3.metric("Margen Real Prom.", f"{margen_prom:,.1f}%" if pd.notna(margen_prom) else "N/D")
col4.metric("Ticket Promedio", f"${ticket_promedio:,.2f}")
col5.metric("Pagos Pendientes", f"${monto_pendiente:,.2f}", delta=f"-${monto_pendiente:,.2f}" if monto_pendiente > 0 else None, delta_color="inverse")

st.markdown("---")

# ─────────────────────────────────────────────
# SECCIÓN 1: Evolución Temporal
# ─────────────────────────────────────────────
st.subheader("📅 Evolución Temporal")

sec1_col1, sec1_col2 = st.columns(2)

# Gráfico YoY — Comparativa Año a Año
if 'Año' in df_filtrado.columns and 'Mes Num' in df_filtrado.columns:
    df_yoy = (
        df_filtrado.groupby(['Año', 'Mes Num', 'Mes Nombre'])['Precio total pagado']
        .sum()
        .reset_index()
        .sort_values(['Año', 'Mes Num'])
    )
    orden_meses = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    df_yoy['Año'] = df_yoy['Año'].astype(str)
    fig_yoy = px.line(
        df_yoy, x='Mes Nombre', y='Precio total pagado', color='Año', markers=True,
        title='Comparativa de Ventas por Año (mismos meses)',
        category_orders={'Mes Nombre': orden_meses},
        labels={'Precio total pagado': 'Ventas ($)', 'Mes Nombre': 'Mes'},
    )
    sec1_col1.plotly_chart(fig_yoy, use_container_width=True)

# Gráfico mensual — Ventas y Comisiones
if 'Mes' in df_filtrado.columns:
    ventas_mes = df_filtrado.groupby('Mes')[['Precio total pagado', 'Comisión Total']].sum().reset_index().sort_values('Mes')
    fig_ventas = go.Figure()
    fig_ventas.add_trace(go.Bar(x=ventas_mes['Mes'], y=ventas_mes['Precio total pagado'], name='Ventas', marker_color='#1f77b4'))
    fig_ventas.add_trace(go.Scatter(x=ventas_mes['Mes'], y=ventas_mes['Comisión Total'], name='Comisiones', mode='lines+markers', yaxis='y2', marker_color='#ff7f0e'))
    fig_ventas.update_layout(
        title='Ventas y Comisiones por Mes',
        yaxis2=dict(title='Comisiones ($)', overlaying='y', side='right'),
        hovermode='x unified',
    )
    sec1_col2.plotly_chart(fig_ventas, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────
# SECCIÓN 2: Mix de Negocio
# ─────────────────────────────────────────────
st.subheader("🗺️ Mix de Negocio")

sec2_col1, sec2_col2 = st.columns(2)

# Treemap — Market Share por Aerolínea (nombres legibles)
if 'Aerolínea Nombre' in df_filtrado.columns:
    fig_tree = px.treemap(
        df_filtrado, path=['Aerolínea Nombre'], values='Precio total pagado',
        title='Market Share por Aerolínea (Volumen de Ventas)',
        hover_data=['Aerolíneas'],
    )
    sec2_col1.plotly_chart(fig_tree, use_container_width=True)

# Top 10 Rutas por Rentabilidad
if 'Comisión Total' in df_filtrado.columns and 'Rutas' in df_filtrado.columns:
    top_rutas = df_filtrado.groupby('Rutas')['Comisión Total'].sum().nlargest(10).reset_index()
    fig_rutas = px.bar(top_rutas, x='Comisión Total', y='Rutas', orientation='h', title='Top 10 Rutas (por Rentabilidad)')
    fig_rutas.update_layout(yaxis={'categoryorder': 'total ascending'})
    sec2_col2.plotly_chart(fig_rutas, use_container_width=True)

# Top 10 Aerolíneas: Ventas (barras) + Margen Real % (línea)
if 'Aerolínea Nombre' in df_filtrado.columns and 'Margen Real %' in df_filtrado.columns:
    agg_aero = (
        df_filtrado.groupby('Aerolínea Nombre')
        .agg(Ventas=('Precio total pagado', 'sum'), Margen=('Margen Real %', 'mean'))
        .nlargest(10, 'Ventas')
        .sort_values('Ventas', ascending=True)
        .reset_index()
    )
    fig_aero = go.Figure()
    fig_aero.add_trace(go.Bar(
        x=agg_aero['Ventas'], y=agg_aero['Aerolínea Nombre'],
        orientation='h', name='Ventas ($)',
        marker_color='#1f77b4',
        hovertemplate='%{y}<br>Ventas: $%{x:,.0f}<extra></extra>',
    ))
    fig_aero.add_trace(go.Scatter(
        x=agg_aero['Margen'], y=agg_aero['Aerolínea Nombre'],
        mode='markers+lines', name='Margen Real %',
        xaxis='x2', marker=dict(color='#ff7f0e', size=10),
        hovertemplate='%{y}<br>Margen: %{x:.1f}%<extra></extra>',
    ))
    fig_aero.update_layout(
        title='Top 10 Aerolíneas: Ventas y Margen Real',
        xaxis=dict(title='Ventas ($)'),
        xaxis2=dict(title='Margen Real (%)', overlaying='x', side='top', showgrid=False),
        legend=dict(orientation='h', yanchor='bottom', y=1.08, xanchor='right', x=1),
        hovermode='y unified',
        margin=dict(t=80),
    )
    st.plotly_chart(fig_aero, use_container_width=True)

# Dona — Nacional vs Internacional
if 'Tipo de Vuelo' in df_filtrado.columns:
    df_tipo = (
        df_filtrado[df_filtrado['Tipo de Vuelo'] != 'Sin clasificar']
        .groupby('Tipo de Vuelo')['Precio total pagado']
        .sum()
        .reset_index()
    )
    _, dona_col, _ = st.columns([1, 2, 1])
    fig_dona = px.pie(
        df_tipo, values='Precio total pagado', names='Tipo de Vuelo',
        hole=0.5,
        title='Segmentación de Ventas: Nacionales vs Internacionales',
        color='Tipo de Vuelo',
        color_discrete_map={'Nacional': '#2ecc71', 'Internacional': '#1f77b4'},
    )
    fig_dona.update_traces(texttemplate='%{label}<br>$%{value:,.0f}<br>(%{percent})', textposition='outside')
    fig_dona.update_layout(showlegend=True)
    dona_col.plotly_chart(fig_dona, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────
# SECCIÓN 3: Rentabilidad
# ─────────────────────────────────────────────
st.subheader("💰 Rentabilidad")

sec3_col1, sec3_col2 = st.columns(2)

# Evolución del Margen Real por Mes
if 'Margen Real %' in df_filtrado.columns and 'Mes' in df_filtrado.columns:
    df_margen = df_filtrado.groupby('Mes')['Margen Real %'].mean().reset_index().sort_values('Mes')
    promedio_global = df_filtrado['Margen Real %'].mean()
    fig_margen = px.line(
        df_margen, x='Mes', y='Margen Real %',
        title='Evolución del Margen Real Promedio por Mes',
        markers=True,
        labels={'Margen Real %': 'Margen Real (%)'},
    )
    fig_margen.add_hline(y=promedio_global, line_dash='dash', annotation_text=f'Promedio: {promedio_global:.1f}%')
    sec3_col1.plotly_chart(fig_margen, use_container_width=True)

# Histograma de distribución de precios
fig_hist = px.histogram(df_filtrado, x='Precio total pagado', nbins=30, title='Distribución de Precios Totales', marginal='box')
sec3_col2.plotly_chart(fig_hist, use_container_width=True)

# ── Scatter de rentabilidad: Rutas y Aerolíneas ──────────────────────────────
if 'Margen Real %' in df_filtrado.columns and 'Margen Real' in df_filtrado.columns:
    avg_margen_global = df_filtrado['Margen Real %'].mean()

    sec3b_col1, sec3b_col2 = st.columns(2)

    # Scatter — Rentabilidad por Ruta (top 5 individuales + "Otros" agrupado)
    if 'Rutas' in df_filtrado.columns:
        df_ruta_r = (
            df_filtrado.groupby('Rutas')
            .agg(
                Ventas=('Precio total pagado', 'sum'),
                Margen_Pct=('Margen Real %', 'mean'),
                Margen_Abs=('Margen Real', 'sum'),
                Pedidos=('Precio total pagado', 'count'),
            )
            .reset_index()
            .sort_values('Ventas', ascending=False)
        )
        top5 = df_ruta_r.head(5).copy()
        resto = df_ruta_r.iloc[5:]
        if not resto.empty:
            otros = pd.DataFrame([{
                'Rutas': 'Otros',
                'Ventas': resto['Ventas'].sum(),
                'Margen_Pct': resto['Margen_Abs'].sum() / resto['Ventas'].sum() * 100 if resto['Ventas'].sum() > 0 else 0,
                'Margen_Abs': resto['Margen_Abs'].sum(),
                'Pedidos': resto['Pedidos'].sum(),
            }])
            df_ruta_r = pd.concat([top5, otros], ignore_index=True)
        else:
            df_ruta_r = top5

        avg_ventas_r = df_ruta_r['Ventas'].mean()

        colores = ['#1f77b4'] * 5 + (['#aec7e8'] if not resto.empty else [])

        fig_ruta_r = px.scatter(
            df_ruta_r,
            x='Ventas', y='Margen_Pct',
            size='Pedidos', text='Rutas',
            hover_data={'Margen_Abs': ':,.0f', 'Pedidos': True, 'Ventas': ':,.0f'},
            title='Rentabilidad por Ruta<br><sup>Top 5 + resto agrupado en "Otros" · tamaño = nº pedidos</sup>',
            labels={'Ventas': 'Ventas ($)', 'Margen_Pct': 'Margen Real (%)', 'Pedidos': 'Pedidos'},
            size_max=55,
        )
        fig_ruta_r.update_traces(textposition='top center', textfont_size=10,
                                 marker=dict(color=colores, opacity=0.85))
        fig_ruta_r.add_vline(x=avg_ventas_r, line_dash='dot', line_color='gray',
                             annotation_text='Prom. ventas', annotation_position='bottom right')
        fig_ruta_r.add_hline(y=avg_margen_global, line_dash='dot', line_color='gray',
                             annotation_text=f'Prom. margen {avg_margen_global:.1f}%', annotation_position='top left')
        fig_ruta_r.update_layout(height=500, showlegend=False)
        sec3b_col1.plotly_chart(fig_ruta_r, use_container_width=True)

    # Scatter — Rentabilidad por Aerolínea (todas)
    if 'Aerolínea Nombre' in df_filtrado.columns:
        df_aero_r = (
            df_filtrado.groupby('Aerolínea Nombre')
            .agg(
                Ventas=('Precio total pagado', 'sum'),
                Margen_Pct=('Margen Real %', 'mean'),
                Margen_Abs=('Margen Real', 'sum'),
                Pedidos=('Precio total pagado', 'count'),
            )
            .reset_index()
        )
        avg_ventas_a = df_aero_r['Ventas'].mean()

        fig_aero_r = px.scatter(
            df_aero_r,
            x='Ventas', y='Margen_Pct',
            size='Pedidos', text='Aerolínea Nombre',
            color='Aerolínea Nombre',
            hover_data={'Margen_Abs': ':,.0f', 'Pedidos': True, 'Ventas': ':,.0f'},
            title='Rentabilidad por Aerolínea<br><sup>Todas las aerolíneas · tamaño = nº pedidos</sup>',
            labels={'Ventas': 'Ventas ($)', 'Margen_Pct': 'Margen Real (%)', 'Pedidos': 'Pedidos'},
            size_max=55,
        )
        fig_aero_r.update_traces(textposition='top center', textfont_size=9)
        fig_aero_r.add_vline(x=avg_ventas_a, line_dash='dot', line_color='gray',
                             annotation_text='Prom. ventas', annotation_position='bottom right')
        fig_aero_r.add_hline(y=avg_margen_global, line_dash='dot', line_color='gray',
                             annotation_text=f'Prom. margen {avg_margen_global:.1f}%', annotation_position='top left')
        fig_aero_r.update_layout(height=500, showlegend=False)
        sec3b_col2.plotly_chart(fig_aero_r, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────
# SECCIÓN 4: Gestión Comercial
# ─────────────────────────────────────────────
st.subheader("👥 Gestión Comercial")

if 'Responsable' in df_filtrado.columns:
    df_resp = df_filtrado[df_filtrado['Responsable'].notna()]
    if not df_resp.empty:
        agg_dict = {'Ventas': ('Precio total pagado', 'sum'), 'Pedidos': ('N. DE PEDIDO', 'count')}
        if 'Margen Real %' in df_resp.columns:
            agg_dict['Margen_Prom'] = ('Margen Real %', 'mean')
        top_resp = df_resp.groupby('Responsable').agg(**agg_dict).nlargest(10, 'Ventas').reset_index()
        hover_cols = ['Pedidos', 'Margen_Prom'] if 'Margen_Prom' in top_resp.columns else ['Pedidos']
        fig_resp = px.bar(
            top_resp, x='Ventas', y='Responsable', orientation='h',
            title='Top 10 Responsables por Volumen de Ventas',
            hover_data=hover_cols,
        )
        fig_resp.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_resp, use_container_width=True)
    else:
        st.info("El 62% de registros no tienen responsable asignado en el Excel de origen. No hay datos suficientes para el ranking.")
else:
    st.info("Columna 'Responsable' no disponible en los datos.")

st.markdown("---")

# ─────────────────────────────────────────────
# SECCIÓN 5: Operativo — Pagos Pendientes
# ─────────────────────────────────────────────
st.subheader("⚠️ Gestión Operativa: Pagos Pendientes")

if 'Estado de Pago' in df_filtrado.columns:
    df_pendientes = df_filtrado[df_filtrado['Estado de Pago'] == 'Pendiente']
    cant_pendiente = len(df_pendientes)
    monto_pend = df_pendientes['Precio total pagado'].sum()

    if not df_pendientes.empty:
        st.error(f"⚠️ {cant_pendiente} pagos pendientes por un total de ${monto_pend:,.2f}")
        cols_tabla = [c for c in ['Fecha de pedido', 'Rutas', 'Aerolíneas', 'Responsable', 'Precio total pagado', 'Comisión Total'] if c in df_pendientes.columns]
        st.dataframe(
            df_pendientes[cols_tabla].sort_values('Precio total pagado', ascending=False),
            use_container_width=True,
        )
    else:
        st.success("¡Excelente! No hay pagos pendientes en la selección actual.")
