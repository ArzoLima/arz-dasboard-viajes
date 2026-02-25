from pathlib import Path, PureWindowsPath
import re
import unicodedata

import pandas as pd


ARCHIVOS_POR_ANIO = {
    2024: r"c:\Users\MBORJA\OneDrive - ARZOBISPADO DE LIMA\IA\Equipos\7_viajes\CUADRO GENERAL DE VENTAS 2024 (P.O.).xlsx",
    2025: r"c:\Users\MBORJA\OneDrive - ARZOBISPADO DE LIMA\IA\Equipos\7_viajes\CUADRO GENERAL DE VENTAS 2025 (P.O.).xlsx",
    2026: r"c:\Users\MBORJA\OneDrive - ARZOBISPADO DE LIMA\IA\Equipos\7_viajes\CUADRO GENERAL DE VENTAS 2026 (P.O.).xlsx",
}

SALIDA_CSV = Path("ventas_viajes.csv")

# Nombres canónicos tal como aparecen en el Excel
COLUMNAS_EXCEL_CANONICAS = [
    "N°",
    "N. DE PEDIDO",
    "FECHA DE PEDIDO",
    "APELLIDOS Y NOMBRE",
    "LÍNEA AÉREA",
    "RUTA",
    "PAGO DEL PASAJERO",
    "PAGO AL PROVEEDOR",
    "FECHA DE ELABORACION DE RECIBO DEL PAGO DEL PASAJERO",
    "FECHA DE PAGO AL PROVEEDOR",
    "FEE ARZOBISPADO",
    "COMISIÓN PROVEEDOR",
    "SUPERIOR - RESPONSABLE DE LA CONGREGACION QUE GARANTIZA EL PAGO AL ARZOBISPADO.",
]

# Contrato de columnas del CSV de salida (consumido por app.py)
COLUMNAS_APP = [
    "correlativo_anual",
    "nro_pedido",
    "fecha_pedido",
    "apellidos_nombres",
    "linea_aerea",
    "ruta",
    "pago_del_pasajero",
    "pago_al_proveedor",
    "fecha_elaboracion_recibo_pasajero",
    "fecha_pago_al_proveedor",
    "fee",
    "comision_proveedor",
    "superior_responsable",
    "estado_pago",
]

# Alias para reconocer encabezados del Excel aunque tengan variantes/errores
ALIAS_EXCEL = {
    "N°": ["n", "n°", "n o", "numero"],
    "N. DE PEDIDO": ["n de pedido", "n. de pedido", "numero de pedido", "pedido"],
    "FECHA DE PEDIDO": ["fecha de pedido", "fecha pedido"],
    "APELLIDOS Y NOMBRE": ["apellidos y nombre", "pasajero", "nombres y apellidos"],
    "LÍNEA AÉREA": ["linea aerea", "línea aérea", "aerolinea", "aerolínea"],
    "RUTA": ["ruta", "rutas", "trayecto"],
    "PAGO DEL PASAJERO": ["pago del pasajero", "total pagado", "precio total", "monto total"],
    "PAGO AL PROVEEDOR": ["pago al proveedor", "costo proveedor"],
    "FECHA DE ELABORACION DE RECIBO DEL PAGO DEL PASAJERO": [
        "fecha de elaboracion de recibo del pago del pasajero",
        "fecha elaboracion recibo",
    ],
    "FECHA DE PAGO AL PROVEEDOR": ["fecha de pago al proveedor", "fecha pago proveedor", "fecha de pago"],
    "FEE ARZOBISPADO": ["fee arzobispado", "fee institucional", "fee"],
    "COMISIÓN PROVEEDOR": ["comision proveedor", "comisión proveedor", "comision"],
    "SUPERIOR - RESPONSABLE DE LA CONGREGACION QUE GARANTIZA EL PAGO AL ARZOBISPADO.": [
        "superior responsable de la congregacion que garantiza el pago al arzobispado",
        "superior responsable de la comgregacion que garantiza el pago al arzobipado",
        "superior responsable",
        "responsable",
    ],
}

# Mapeo: nombre canónico del Excel → nombre snake_case del CSV
MAPEO_EXCEL_A_APP = {
    "N°": "correlativo_anual",
    "N. DE PEDIDO": "nro_pedido",
    "FECHA DE PEDIDO": "fecha_pedido",
    "APELLIDOS Y NOMBRE": "apellidos_nombres",
    "LÍNEA AÉREA": "linea_aerea",
    "RUTA": "ruta",
    "PAGO DEL PASAJERO": "pago_del_pasajero",
    "PAGO AL PROVEEDOR": "pago_al_proveedor",
    "FECHA DE ELABORACION DE RECIBO DEL PAGO DEL PASAJERO": "fecha_elaboracion_recibo_pasajero",
    "FECHA DE PAGO AL PROVEEDOR": "fecha_pago_al_proveedor",
    "FEE ARZOBISPADO": "fee",
    "COMISIÓN PROVEEDOR": "comision_proveedor",
    "SUPERIOR - RESPONSABLE DE LA CONGREGACION QUE GARANTIZA EL PAGO AL ARZOBISPADO.": "superior_responsable",
}


def normalizar_texto(valor: str) -> str:
    """Normaliza texto para comparar encabezados de forma robusta."""
    txt = str(valor).strip().lower()
    txt = unicodedata.normalize("NFKD", txt)
    txt = "".join(c for c in txt if not unicodedata.combining(c))
    txt = re.sub(r"[^a-z0-9]+", " ", txt).strip()
    return txt


def ruta_windows_a_wsl(ruta: str) -> Path:
    """Convierte 'C:\\...' a '/mnt/c/...' cuando corresponde."""
    p = PureWindowsPath(ruta)
    if p.drive:
        letra = p.drive.replace(":", "").lower()
        return Path("/mnt") / letra / Path(*p.parts[1:])
    return Path(ruta)


def _detectar_fila_cabecera(raw: pd.DataFrame) -> int:
    """Busca la fila que contiene la cabecera real del cuadro."""
    for i in range(min(len(raw), 40)):
        fila = [normalizar_texto(v) for v in raw.iloc[i].tolist()]
        if "n de pedido" in fila and "fecha de pedido" in fila and "ruta" in fila:
            return i
    raise ValueError("No se encontró una cabecera válida en el Excel.")


def asociar_columnas_excel(df: pd.DataFrame) -> pd.DataFrame:
    """Asocia columnas del formato Excel a nombres canónicos del cuadro."""
    columnas_actuales_norm = {normalizar_texto(col): col for col in df.columns}
    renombres: dict[str, str] = {}

    for col_objetivo, aliases in ALIAS_EXCEL.items():
        candidatos = [col_objetivo, *aliases]
        for candidato in candidatos:
            candidato_norm = normalizar_texto(candidato)
            if candidato_norm in columnas_actuales_norm:
                col_origen = columnas_actuales_norm[candidato_norm]
                renombres[col_origen] = col_objetivo
                break

    df = df.rename(columns=renombres)
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    presentes = [c for c in COLUMNAS_EXCEL_CANONICAS if c in df.columns]
    return df[presentes].copy()


def convertir_columnas_app(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte columnas canónicas de Excel a nombres snake_case del CSV."""
    df = df.rename(columns=MAPEO_EXCEL_A_APP)

    if "linea_aerea" in df.columns:
        df["linea_aerea"] = df["linea_aerea"].str.strip()

    if "estado_pago" not in df.columns:
        if "fecha_pago_al_proveedor" in df.columns:
            fecha_pago = pd.to_datetime(df["fecha_pago_al_proveedor"], errors="coerce")
            df["estado_pago"] = fecha_pago.apply(lambda x: "Pendiente" if pd.isna(x) else "Pagado")
        else:
            df["estado_pago"] = "Pendiente"

    for col_numerica in ["fee", "comision_proveedor", "pago_del_pasajero", "pago_al_proveedor"]:
        if col_numerica in df.columns:
            df[col_numerica] = pd.to_numeric(df[col_numerica], errors="coerce")

    return df


def leer_excel_con_anio(ruta_original: str, anio: int) -> pd.DataFrame:
    ruta = ruta_windows_a_wsl(ruta_original)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {ruta}")

    raw = pd.read_excel(ruta, header=None)
    fila_cabecera = _detectar_fila_cabecera(raw)
    headers = raw.iloc[fila_cabecera].astype(str).str.strip()

    df = raw.iloc[fila_cabecera + 1 :].copy()
    df.columns = headers
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    df = df.dropna(how="all")
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed", na=False)]

    df = asociar_columnas_excel(df)
    df = convertir_columnas_app(df)
    df["anio"] = anio
    return df


def completar_columnas_faltantes(df: pd.DataFrame) -> pd.DataFrame:
    """Asegura que todas las columnas que consume app.py existan."""
    for col in COLUMNAS_APP:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def main() -> None:
    dataframes = []

    for anio, ruta in ARCHIVOS_POR_ANIO.items():
        df = leer_excel_con_anio(ruta, anio)
        dataframes.append(df)

    df_final = pd.concat(dataframes, ignore_index=True)
    df_final = completar_columnas_faltantes(df_final)

    orden_final = [*COLUMNAS_APP, "anio"]
    columnas_restantes = [c for c in df_final.columns if c not in orden_final]
    df_final = df_final[orden_final + columnas_restantes]

    df_final.to_csv(SALIDA_CSV, index=False, encoding="utf-8-sig")

    print(f"CSV generado: {SALIDA_CSV.resolve()}")
    print(f"Filas totales: {len(df_final):,}")


if __name__ == "__main__":
    main()
