import os
import time
import warnings
from datetime import datetime, timedelta, timezone
from threading import Thread

# Silenciar mensajes informativos y advertencias de TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

from flask import Flask
import numpy as np
import pandas as pd
import requests
from scipy.stats import poisson
import telebot
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Model
from xgboost import XGBClassifier, XGBRegressor

# ==========================================
# SERVIDOR FLASK OPTIMIZADO PARA RENDER
# ==========================================
app = Flask("")


@app.route("/")
def home():
    return "🤖 Bot de Predicciones de Fútbol Activo 24/7", 200


def run_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)


# ==========================================
# CONFIGURACIÓN DE CREDENCIALES Y DICCIONARIOS
# ==========================================
API_KEY = "40f348e5bd5646daa606c0096cb13fb2"
TELEGRAM_TOKEN = "8708621550:AAHHS-YQaABZWILyQyudJ53FX8gOjNl7hBs"

bot = telebot.TeleBot(TELEGRAM_TOKEN)
HEADERS = {"X-Auth-Token": API_KEY}
SEASONS_PAST = [2021, 2022, 2023, 2024, 2025]
CURRENT_SEASON = 2026

# Sesión reutilizable para mayor velocidad de peticiones
session = requests.Session()
session.headers.update(HEADERS)

LIGAS_DISPONIBLES = {
    "CL": "Champions League",
    "PD": "Primera División (España)",
    "PL": "Premier League (Inglaterra)",
    "BL1": "Bundesliga (Alemania)",
    "SA": "Serie A (Italia)",
    "FL1": "Ligue 1 (Francia)",
    "DED": "Eredivisie (Países Bajos)",
    "PPL": "Primeira Liga (Portugal)",
    "ELC": "Championship (Inglaterra)",
}


# ==========================================
# FUNCIONES AUXILIARES
# ==========================================
def enviar_mensaje_largo(chat_id, texto, limite_caracteres=3800):
    """Divide mensajes largos para no exceder el límite de Telegram."""
    if len(texto) <= limite_caracteres:
        partes = [texto]
    else:
        partes = []
        bloques = texto.split("─────────────────────────────\n")
        parte_actual = ""

        for bloque in bloques:
            if len(parte_actual) + len(bloque) + 30 > limite_caracteres:
                partes.append(parte_actual)
                parte_actual = bloque + "─────────────────────────────\n"
            else:
                parte_actual += bloque + "─────────────────────────────\n"

        if parte_actual.strip():
            partes.append(parte_actual)

    for p in partes:
        if p.strip():
            try:
                bot.send_message(chat_id, p, parse_mode="Markdown")
            except Exception:
                bot.send_message(chat_id, p)
            time.sleep(0.3)


# ==========================================
# GESTIÓN DE DATOS Y CACHÉ LOCAL EN CSV
# ==========================================
def get_historical_data_cached(league_code):
    """Carga temporadas pasadas desde CSV local (o descarga 1 vez) y concatena la temporada actual."""
    csv_file = f"cache_{league_code}_historical.csv"
    df_past = pd.DataFrame()

    # 1. Cargar o crear caché de temporadas pasadas
    if os.path.exists(csv_file):
        try:
            df_past = pd.read_csv(csv_file)
            df_past["utcDate"] = pd.to_datetime(df_past["utcDate"])
        except Exception as e:
            print(f"Error leyendo CSV local de {league_code}: {e}")

    if df_past.empty:
        past_matches = []
        for s in SEASONS_PAST:
            url = f"https://api.football-data.org/v4/competitions/{league_code}/matches?season={s}&status=FINISHED"
            try:
                res = session.get(url)
                if res.status_code == 200:
                    data = res.json().get("matches", [])
                    for m in data:
                        past_matches.append(
                            {
                                "season": s,
                                "utcDate": m.get("utcDate"),
                                "home": m["homeTeam"]["name"],
                                "away": m["awayTeam"]["name"],
                                "h_g": m["score"]["fullTime"]["home"],
                                "a_g": m["score"]["fullTime"]["away"],
                            }
                        )
                time.sleep(0.2)
            except Exception as e:
                print(f"Error descargando temporada {s}: {e}")

        df_past = pd.DataFrame(past_matches)
        if not df_past.empty:
            df_past.to_csv(csv_file, index=False)
            df_past["utcDate"] = pd.to_datetime(df_past["utcDate"])

    # 2. Descargar únicamente la temporada actual (2026)
    current_matches = []
    url_curr = f"https://api.football-data.org/v4/competitions/{league_code}/matches?season={CURRENT_SEASON}&status=FINISHED"
    try:
        res = session.get(url_curr)
        if res.status_code == 200:
            data = res.json().get("matches", [])
            for m in data:
                current_matches.append(
                    {
                        "season": CURRENT_SEASON,
                        "utcDate": m.get("utcDate"),
                        "home": m["homeTeam"]["name"],
                        "away": m["awayTeam"]["name"],
                        "h_g": m["score"]["fullTime"]["home"],
                        "a_g": m["score"]["fullTime"]["away"],
                    }
                )
    except Exception as e:
        print(f"Error actualizando temporada actual {league_code}: {e}")

    df_curr = pd.DataFrame(current_matches)
    if not df_curr.empty:
        df_curr["utcDate"] = pd.to_datetime(df_curr["utcDate"])
        df_full = pd.concat([df_past, df_curr], ignore_index=True)
    else:
        df_full = df_past

    if not df_full.empty:
        df_full = df_full.sort_values("utcDate").reset_index(drop=True)

    return df_full


# ==========================================
# INGENIERÍA DE CARACTERÍSTICAS AVANZADA
# ==========================================
def get_team_history_advanced(df, team_name, match_date, window=5):
    """Extrae características avanzadas: forma ponderada, rendimiento específico y días de descanso."""
    df_prev = df[df["utcDate"] < match_date]
    team_matches = df_prev[
        (df_prev["home"] == team_name) | (df_prev["away"] == team_name)
    ].copy()

    if len(team_matches) < window:
        return np.ones((window, 2)), 1.0, 1.0, 7.0  # (seq, gf, ga, rest_days)

    recent = team_matches.tail(window)
    feats = []
    gf_list, ga_list = [], []

    for _, row in recent.iterrows():
        if row["home"] == team_name:
            gf_list.append(row["h_g"])
            ga_list.append(row["a_g"])
            feats.append([row["h_g"], row["a_g"]])
        else:
            gf_list.append(row["a_g"])
            ga_list.append(row["h_g"])
            feats.append([row["a_g"], row["h_g"]])

    # 1. Ponderación exponencial (los partidos más recientes valen más)
    weights = np.exp(np.linspace(-1, 0, window))
    weights /= weights.sum()

    weighted_gf = np.dot(gf_list, weights)
    weighted_ga = np.dot(ga_list, weights)

    # 2. Días de descanso desde el último partido
    last_match_date = team_matches.iloc[-1]["utcDate"]
    rest_days = (
        (match_date - last_match_date).total_seconds() / 86400.0
        if pd.notnull(last_match_date)
        else 7.0
    )
    rest_days = min(15.0, max(2.0, rest_days))  # Acotar entre 2 y 15 días

    return np.array(feats), weighted_gf, weighted_ga, rest_days


def get_venue_specific_stats(df, home_team, away_team, match_date, window=5):
    """Obtiene el rendimiento específico de Local en casa y Visitante fuera."""
    df_prev = df[df["utcDate"] < match_date]

    home_matches = df_prev[df_prev["home"] == home_team].tail(window)
    away_matches = df_prev[df_prev["away"] == away_team].tail(window)

    h_gf = (
        home_matches["h_g"].mean()
        if len(home_matches) > 0
        else df_prev["h_g"].mean()
    )
    h_ga = (
        home_matches["a_g"].mean()
        if len(home_matches) > 0
        else df_prev["a_g"].mean()
    )

    a_gf = (
        away_matches["a_g"].mean()
        if len(away_matches) > 0
        else df_prev["a_g"].mean()
    )
    a_ga = (
        away_matches["h_g"].mean()
        if len(away_matches) > 0
        else df_prev["h_g"].mean()
    )

    return h_gf, h_ga, a_gf, a_ga


def prepare_engine(df, window=5):
    X_seq, X_stat, y_1x2, y_corners_target = [], [], [], []
    mean_h_g = df["h_g"].mean() if len(df) > 0 else 1.4
    mean_a_g = df["a_g"].mean() if len(df) > 0 else 1.1

    for i in range(window * 2, len(df)):
        h_team, a_team = df.iloc[i]["home"], df.iloc[i]["away"]
        m_date = df.iloc[i]["utcDate"]

        r_h, h_gf_w, h_ga_w, h_rest = get_team_history_advanced(
            df, h_team, m_date, window=window
        )
        r_a, a_gf_w, a_ga_w, a_rest = get_team_history_advanced(
            df, a_team, m_date, window=window
        )

        vh_gf, vh_ga, va_gf, va_ga = get_venue_specific_stats(
            df, h_team, a_team, m_date, window=window
        )

        seq = np.hstack([r_h, r_a])

        hg, ag = df.iloc[i]["h_g"], df.iloc[i]["a_g"]
        target_1x2 = 0 if hg > ag else (1 if hg == ag else 2)

        # Proxy de córners históricos según intensidad
        corners_proxy = (h_gf_w + a_gf_w) * 2.1 + (h_ga_w + a_ga_w) * 1.5 + 4.0

        X_seq.append(seq)
        X_stat.append(
            [
                h_gf_w,
                h_ga_w,
                a_gf_w,
                a_ga_w,
                vh_gf,
                vh_ga,
                va_gf,
                va_ga,
                h_rest,
                a_rest,
                mean_h_g,
                mean_a_g,
            ]
        )
        y_1x2.append(target_1x2)
        y_corners_target.append(corners_proxy)

    return (
        np.array(X_seq),
        np.array(X_stat),
        np.array(y_1x2),
        np.array(y_corners_target),
    )


# ==========================================
# CÁLCULOS MATEMÁTICOS DE POISSON
# ==========================================
def calcular_probabilidades_corners(
    lambda_corners, lineas=[7.5, 8.5, 9.5, 10.5, 11.5]
):
    probs = {}
    for linea in lineas:
        k_max = int(np.floor(linea))
        prob_under = sum(
            poisson.pmf(k, lambda_corners) for k in range(k_max + 1)
        )
        probs[f"+{linea}"] = (1 - prob_under) * 100
    return probs


def calcular_probabilidades_poisson(lambda_home, lambda_away, max_goles=6):
    prob_matrix = np.zeros((max_goles, max_goles))
    for i in range(max_goles):
        for j in range(max_goles):
            prob_matrix[i, j] = poisson.pmf(i, lambda_home) * poisson.pmf(
                j, lambda_away
            )

    p_home = np.sum(np.tril(prob_matrix, -1))
    p_draw = np.sum(np.diag(prob_matrix))
    p_away = np.sum(np.triu(prob_matrix, 1))

    total = p_home + p_draw + p_away
    return p_home / total, p_draw / total, p_away / total


# ==========================================
# GENERACIÓN DE REPORTE Y PREDICCIÓN CON IA
# ==========================================
def ejecutar_modelo_y_generar_reporte(league_code, dias_a_futuro=7):
    df = get_historical_data_cached(league_code)
    if df.empty or "home" not in df.columns:
        return f"❌ No se pudieron extraer datos históricos para *{league_code}*."

    X_s, X_st, y_train, y_corners = prepare_engine(df)
    if len(X_s) == 0:
        return (
            f"⚠️ No hay suficientes datos procesables para *{league_code}*."
        )

    # 1. Modelo LSTM para extractor de secuencias
    inp = Input(shape=(5, 4))
    x = LSTM(32, return_sequences=False)(inp)
    x = Dropout(0.2)(x)
    fe = Dense(16, activation="relu")(x)
    out = Dense(3, activation="softmax")(fe)

    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(X_s, y_train, epochs=12, batch_size=32, verbose=0)

    # 2. Extractor de características + XGBoost Classifier (1X2)
    extractor = Model(inputs=inp, outputs=fe)
    extracted_features = extractor.predict(X_s, verbose=0)
    X_final = np.hstack([extracted_features, X_st])

    xgb_1x2 = XGBClassifier(
        n_estimators=100,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    ).fit(X_final, y_train)

    # 3. XGBoost Regressor para Córners
    xgb_corners = XGBRegressor(
        n_estimators=80, learning_rate=0.04, max_depth=3, random_state=42
    ).fit(X_st, y_corners)

    # 4. Consulta de Próximos Partidos
    url = f"https://api.football-data.org/v4/competitions/{league_code}/matches?status=SCHEDULED,IN_PLAY,PAUSED"
    res = session.get(url)
    if res.status_code != 200:
        return f"⚠️ Error consultando partidos ({res.status_code})."

    proximos = res.json().get("matches", [])
    if not proximos:
        nombre_liga = LIGAS_DISPONIBLES.get(league_code, league_code)
        return (
            f"⚠️ No hay encuentros programados ni en vivo para *{nombre_liga}*."
        )

    ahora = datetime.now(timezone.utc)
    limite_fecha = ahora + timedelta(days=dias_a_futuro)

    partidos_filtrados = []
    for m in proximos:
        fecha_match = datetime.fromisoformat(
            m["utcDate"].replace("Z", "+00:00")
        )
        if m.get("status") in ["IN_PLAY", "PAUSED"] or (
            ahora <= fecha_match <= limite_fecha
        ):
            partidos_filtrados.append(m)

    if not partidos_filtrados:
        proximos = sorted(proximos, key=lambda x: x["utcDate"])
        partidos_filtrados = proximos[:10]
    else:
        partidos_filtrados = sorted(
            partidos_filtrados, key=lambda x: x["utcDate"]
        )

    proximos = partidos_filtrados

    # 5. Formatear reporte
    nombre_liga = LIGAS_DISPONIBLES.get(league_code, league_code)
    reporte = f"📊 *REPORTE ANALÍTICO PRO: {nombre_liga}*\n"
    reporte += "═" * 30 + "\n\n"

    mean_h_g = df["h_g"].mean() if len(df) > 0 else 1.4
    mean_a_g = df["a_g"].mean() if len(df) > 0 else 1.1

    for m in proximos:
        h_name, a_name = m["homeTeam"]["name"], m["awayTeam"]["name"]
        fecha = m["utcDate"][:10]
        match_date = pd.to_datetime(m["utcDate"])
        jornada = m.get("matchday", "N/A")
        estado_str = (
            "🔴 EN VIVO"
            if m.get("status") in ["IN_PLAY", "PAUSED"]
            else f"📅 JORNADA {jornada}"
        )

        r_h, h_gf, h_ga, h_rest = get_team_history_advanced(
            df, h_name, match_date, window=5
        )
        r_a, a_gf, a_ga, a_rest = get_team_history_advanced(
            df, a_name, match_date, window=5
        )

        vh_gf, vh_ga, va_gf, va_ga = get_venue_specific_stats(
            df, h_name, a_name, match_date, window=5
        )

        avg_h_xg = max(
            0.2, (vh_gf / mean_h_g) * (va_ga / mean_a_g) * mean_h_g
        )
        avg_a_xg = max(
            0.2, (va_gf / mean_a_g) * (vh_ga / mean_h_g) * mean_a_g
        )
        total_xg = avg_h_xg + avg_a_xg

        seq = np.hstack([r_h, r_a]).reshape(1, 5, 4)
        feat = extractor.predict(seq, verbose=0)
        static_feats = np.array(
            [[
                h_gf,
                h_ga,
                a_gf,
                a_ga,
                vh_gf,
                vh_ga,
                va_gf,
                va_ga,
                h_rest,
                a_rest,
                mean_h_g,
                mean_a_g,
            ]]
        )

        prob_ia = xgb_1x2.predict_proba(np.hstack([feat, static_feats]))[0]

        p_h_stat, p_e_stat, p_v_stat = calcular_probabilidades_poisson(
            avg_h_xg, avg_a_xg
        )

        p_h = 0.6 * prob_ia[0] + 0.4 * p_h_stat
        p_e = 0.6 * prob_ia[1] + 0.4 * p_e_stat
        p_v = 0.6 * prob_ia[2] + 0.4 * p_v_stat

        total_p = p_h + p_e + p_v
        p_h, p_e, p_v = p_h / total_p, p_e / total_p, p_v / total_p

        p_btts = (
            (1 - poisson.pmf(0, avg_h_xg))
            * (1 - poisson.pmf(0, avg_a_xg))
            * 100
        )
        p_over25 = (1 - sum(poisson.pmf(i, total_xg) for i in range(3))) * 100

        # Predicción de Córners mediante XGBRegressor
        total_exp_corners = float(xgb_corners.predict(static_feats)[0])
        exp_c_home = total_exp_corners * (
            avg_h_xg / max(0.1, (avg_h_xg + avg_a_xg))
        )
        exp_c_away = total_exp_corners - exp_c_home

        probs_corners = calcular_probabilidades_corners(total_exp_corners)

        # Construcción del mensaje
        reporte += f"{estado_str} | 📅 {fecha}\n"
        reporte += f"⚽ *{h_name} vs {a_name}*\n"
        reporte += f"⏳ Descanso: {h_name} ({h_rest:.0f}d) | {a_name} ({a_rest:.0f}d)\n"
        reporte += f"🏆 Local: {p_h:.1%} | Empate: {p_e:.1%} | Visita: {p_v:.1%}\n"
        reporte += f"🥅 Goles Est.: {h_name}: {avg_h_xg:.2f} | {a_name}: {avg_a_xg:.2f}\n"
        reporte += f"🔥 Both Score: {p_btts:.1f}% | Over 2.5 Goles: {p_over25:.1f}%\n"
        reporte += f"🚩 *Córners Estimados:* {h_name}: {exp_c_home:.1f} | {a_name}: {exp_c_away:.1f} (Total: {total_exp_corners:.1f})\n"
        reporte += f"📈 *Probabilidades Córners:*\n"
        reporte += f"    • +7.5: {probs_corners['+7.5']:.1f}% | +8.5: {probs_corners['+8.5']:.1f}%\n"
        reporte += f"    • +9.5: {probs_corners['+9.5']:.1f}% | +10.5: {probs_corners['+10.5']:.1f}%\n"
        reporte += f"    • +11.5: {probs_corners['+11.5']:.1f}%\n"
        reporte += "─────────────────────────────\n"

    return reporte


# ==========================================
# HANDLERS DEL BOT
# ==========================================
def obtener_menu_ligas():
    menu = "⚽ *LIGAS DISPONIBLES EN EL BOT*\n\n"
    menu += "Envía el código de la liga para generar el análisis:\n\n"
    for codigo, nombre in LIGAS_DISPONIBLES.items():
        menu += f"• *{codigo}* ➔ {nombre}\n"
    return menu


@bot.message_handler(commands=["start", "help"])
def enviar_bienvenida(message):
    bienvenida = (
        "🤖 *Bot de Predicción Predictiva de Fútbol Pro*\n"
        "Combinación de Deep Learning (LSTM), XGBoost (1X2 y Córners), Poisson y Métricas de Descanso.\n\n"
    )
    bot.reply_to(
        message, bienvenida + obtener_menu_ligas(), parse_mode="Markdown"
    )


@bot.message_handler(func=lambda msg: True)
def responder_prompt(message):
    texto = message.text.strip().upper()

    if texto in LIGAS_DISPONIBLES:
        nombre_liga = LIGAS_DISPONIBLES[texto]
        bot.reply_to(
            message,
            f"⏳ Procesando datos e impulsando modelos IA para *{nombre_liga}* ({texto})... por favor espera.",
            parse_mode="Markdown",
        )
        try:
            resultado = ejecutar_modelo_y_generar_reporte(
                texto, dias_a_futuro=7
            )
            enviar_mensaje_largo(message.chat.id, resultado)

        except Exception as e:
            bot.reply_to(
                message, f"❌ Ocurrió un error al procesar la solicitud: {e}"
            )
    else:
        bot.reply_to(
            message,
            f"⚠️ Código de liga no reconocido.\n\n" + obtener_menu_ligas(),
            parse_mode="Markdown",
        )


if __name__ == "__main__":
    Thread(target=run_flask).start()

    print("🤖 Bot de Telegram activo y esperando comandos...")
    bot.infinity_polling(timeout=10, long_polling_timeout=5)
