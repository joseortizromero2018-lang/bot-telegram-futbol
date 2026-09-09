import os
import time
import warnings
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
from xgboost import XGBClassifier

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
TELEGRAM_TOKEN = "8847776327:AAFfy6zP0uS1EwrXbgeR2WlJTNLul7gd50Q"

bot = telebot.TeleBot(TELEGRAM_TOKEN)
HEADERS = {"X-Auth-Token": API_KEY}
SEASONS_TO_FETCH = [2023, 2024, 2025, 2026]

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
# CÁLCULOS DINÁMICOS DE CÓRNERS Y POISSON
# ==========================================
def get_team_corners_history_estimated(
    df, team_name, mean_league_gf, window=5
):
    """Estima los córners proyectados en función de la producción ofensiva y defensiva reciente."""
    team_matches = df[
        (df["home"] == team_name) | (df["away"] == team_name)
    ].copy()

    if len(team_matches) < window:
        return 4.8, 4.7

    recent = team_matches.tail(window)
    gf_list, ga_list = [], []

    for _, row in recent.iterrows():
        if row["home"] == team_name:
            gf_list.append(row["h_g"])
            ga_list.append(row["a_g"])
        else:
            gf_list.append(row["a_g"])
            ga_list.append(row["h_g"])

    avg_gf = np.mean(gf_list)
    avg_ga = np.mean(ga_list)

    ratio_ataque = (
        (avg_gf / max(0.5, mean_league_gf)) if mean_league_gf > 0 else 1.0
    )
    ratio_defensa = (
        (avg_ga / max(0.5, mean_league_gf)) if mean_league_gf > 0 else 1.0
    )

    c_favor_est = max(2.5, min(8.5, 4.75 * (0.6 * ratio_ataque + 0.4)))
    c_contra_est = max(2.5, min(8.5, 4.5 * (0.6 * ratio_defensa + 0.4)))

    return c_favor_est, c_contra_est


def calcular_probabilidades_corners(
    lambda_corners, lineas=[7.5, 8.5, 9.5, 10.5, 11.5]
):
    """Calcula la probabilidad de superar las líneas de córners (+7.5 a +11.5) mediante Poisson."""
    probs = {}
    for linea in lineas:
        k_max = int(np.floor(linea))
        prob_under = sum(
            poisson.pmf(k, lambda_corners) for k in range(k_max + 1)
        )
        probs[f"+{linea}"] = (1 - prob_under) * 100
    return probs


def calcular_probabilidades_poisson(lambda_home, lambda_away, max_goles=6):
    """Calcula probabilidades de resultado 1X2 para goles."""
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
# PROCESAMIENTO DE DATOS HISTÓRICOS
# ==========================================
def get_historical_data_multiseason(league_code, seasons):
    all_matches = []
    for s in seasons:
        url = f"https://api.football-data.org/v4/competitions/{league_code}/matches?season={s}&status=FINISHED"
        try:
            res = requests.get(url, headers=HEADERS)
            if res.status_code == 200:
                data = res.json().get("matches", [])
                for m in data:
                    all_matches.append(
                        {
                            "season": s,
                            "utcDate": m.get("utcDate"),
                            "home": m["homeTeam"]["name"],
                            "away": m["awayTeam"]["name"],
                            "h_g": m["score"]["fullTime"]["home"],
                            "a_g": m["score"]["fullTime"]["away"],
                        }
                    )
            time.sleep(0.3)
        except Exception as e:
            print(f"Error extrayendo temporada {s}: {e}")

    df = pd.DataFrame(all_matches)
    if not df.empty:
        df["utcDate"] = pd.to_datetime(df["utcDate"])
        df = df.sort_values("utcDate").reset_index(drop=True)
    return df


def get_team_history(df, team_name, window=5):
    team_matches = df[
        (df["home"] == team_name) | (df["away"] == team_name)
    ].copy()
    if len(team_matches) < window:
        return np.ones((window, 2))

    recent = team_matches.tail(window)
    feats = []
    for _, row in recent.iterrows():
        if row["home"] == team_name:
            feats.append([row["h_g"], row["a_g"]])
        else:
            feats.append([row["a_g"], row["h_g"]])

    return np.array(feats)


def prepare_engine(df, window=5):
    X_seq, X_stat, y = [], [], []
    mean_h_g = df["h_g"].mean() if len(df) > 0 else 1.4
    mean_a_g = df["a_g"].mean() if len(df) > 0 else 1.1

    for i in range(window * 2, len(df)):
        h_team, a_team = df.iloc[i]["home"], df.iloc[i]["away"]
        df_sub = df.iloc[:i]

        r_h = get_team_history(df_sub, h_team, window=window)
        r_a = get_team_history(df_sub, a_team, window=window)

        seq = np.hstack([r_h, r_a])
        h_gf, h_ga = np.mean(r_h[:, 0]), np.mean(r_h[:, 1])
        a_gf, a_ga = np.mean(r_a[:, 0]), np.mean(r_a[:, 1])

        hg, ag = df.iloc[i]["h_g"], df.iloc[i]["a_g"]
        target = 0 if hg > ag else (1 if hg == ag else 2)

        X_seq.append(seq)
        X_stat.append([h_gf, h_ga, a_gf, a_ga, mean_h_g, mean_a_g])
        y.append(target)

    return np.array(X_seq), np.array(X_stat), np.array(y)


# ==========================================
# GENERACIÓN DE REPORTE CON CÓRNERS DINÁMICOS
# ==========================================
def ejecutar_modelo_y_generar_reporte(league_code, max_jornadas=1):
    df = get_historical_data_multiseason(league_code, SEASONS_TO_FETCH)
    if df.empty or "home" not in df.columns:
        return f"❌ No se pudieron extraer datos históricos para *{league_code}*."

    X_s, X_st, y_train = prepare_engine(df)
    if len(X_s) == 0:
        return (
            f"⚠️ No hay suficientes datos procesables para *{league_code}*."
        )

    # 1. Modelo LSTM
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
    model.fit(X_s, y_train, epochs=15, batch_size=16, verbose=0)

    # 2. Extractor de características + XGBoost
    extractor = Model(inputs=inp, outputs=fe)
    extracted_features = extractor.predict(X_s, verbose=0)
    X_final = np.hstack([extracted_features, X_st])

    xgb = XGBClassifier(
        n_estimators=100,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    ).fit(X_final, y_train)

    # 3. Consulta de Próximos Partidos
    url = f"https://api.football-data.org/v4/competitions/{league_code}/matches?status=SCHEDULED,IN_PLAY,PAUSED"
    res = requests.get(url, headers=HEADERS)
    if res.status_code != 200:
        return f"⚠️ Error consultando partidos ({res.status_code})."

    proximos = res.json().get("matches", [])
    if not proximos:
        nombre_liga = LIGAS_DISPONIBLES.get(league_code, league_code)
        return (
            f"⚠️ No hay encuentros programados ni en vivo para *{nombre_liga}*."
        )

    proximos = sorted(proximos, key=lambda x: x["utcDate"])
    matchdays = sorted(
        list(
            set(
                m.get("matchday")
                for m in proximos
                if m.get("matchday") is not None
            )
        )
    )
    if matchdays:
        proximos = [
            m for m in proximos if m.get("matchday") in matchdays[:max_jornadas]
        ]

    # 4. Formatear reporte con sección de Córners Adaptativos
    nombre_liga = LIGAS_DISPONIBLES.get(league_code, league_code)
    reporte = f"📊 *REPORTE ANALÍTICO PRO: {nombre_liga}*\n"
    reporte += "═" * 30 + "\n\n"

    mean_h_g = df["h_g"].mean() if len(df) > 0 else 1.4
    mean_a_g = df["a_g"].mean() if len(df) > 0 else 1.1

    for m in proximos:
        h_name, a_name = m["homeTeam"]["name"], m["awayTeam"]["name"]
        fecha = m["utcDate"][:10]
        estado_str = (
            "🔴 EN VIVO"
            if m.get("status") in ["IN_PLAY", "PAUSED"]
            else "📅 PROGRAMADO"
        )

        r_h = get_team_history(df, h_name, window=5)
        r_a = get_team_history(df, a_name, window=5)

        h_gf, h_ga = np.mean(r_h[:, 0]), np.mean(r_h[:, 1])
        a_gf, a_ga = np.mean(r_a[:, 0]), np.mean(r_a[:, 1])

        avg_h_xg = max(
            0.2, (h_gf / mean_h_g) * (a_ga / mean_a_g) * mean_h_g
        )
        avg_a_xg = max(
            0.2, (a_gf / mean_a_g) * (h_ga / mean_h_g) * mean_a_g
        )
        total_xg = avg_h_xg + avg_a_xg

        seq = np.hstack([r_h, r_a]).reshape(1, 5, 4)
        feat = extractor.predict(seq, verbose=0)
        static_feats = np.array(
            [[h_gf, h_ga, a_gf, a_ga, mean_h_g, mean_a_g]]
        )
        prob_ia = xgb.predict_proba(np.hstack([feat, static_feats]))[0]

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

        # Cálculo de Córners dinámicos en base al volumen ofensivo
        c_h_favor, c_h_contra = get_team_corners_history_estimated(
            df, h_name, mean_h_g
        )
        c_a_favor, c_a_contra = get_team_corners_history_estimated(
            df, a_name, mean_a_g
        )

        exp_c_home = (c_h_favor + c_a_contra) / 2
        exp_c_away = (c_a_favor + c_h_contra) / 2
        total_exp_corners = exp_c_home + exp_c_away

        probs_corners = calcular_probabilidades_corners(total_exp_corners)

        # Construcción del mensaje
        reporte += f"{estado_str} | 📅 {fecha}\n"
        reporte += f"⚽ *{h_name} vs {a_name}*\n"
        reporte += f"🏆 Local: {p_h:.1%} | Empate: {p_e:.1%} | Visita: {p_v:.1%}\n"
        reporte += f"🥅 Goles Est.: {h_name}: {avg_h_xg:.2f} | {a_name}: {avg_a_xg:.2f}\n"
        reporte += f"🔥 Both Score: {p_btts:.1f}% | Over 2.5 Goles: {p_over25:.1f}%\n"
        reporte += f"🚩 *Córners Estimados:* {h_name}: {exp_c_home:.1f} | {a_name}: {exp_c_away:.1f} (Total: {total_exp_corners:.1f})\n"
        reporte += f"📈 *Probabilidades Córners:*\n"
        reporte += f"   • +7.5: {probs_corners['+7.5']:.1f}% | +8.5: {probs_corners['+8.5']:.1f}%\n"
        reporte += f"   • +9.5: {probs_corners['+9.5']:.1f}% | +10.5: {probs_corners['+10.5']:.1f}%\n"
        reporte += f"   • +11.5: {probs_corners['+11.5']:.1f}%\n"
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
        "🤖 *Bot de Predicción Predictiva de Fútbol*\n"
        "Combinación de Deep Learning (LSTM), XGBoost y Distribución de Poisson.\n\n"
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
                texto, max_jornadas=1
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
