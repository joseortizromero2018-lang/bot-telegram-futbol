import os
import time
import warnings
from datetime import datetime, timedelta, timezone
from threading import Thread

# Silenciar mensajes informativos y advertencias de TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

from flask import Flask
import joblib
import numpy as np
import pandas as pd
import requests
from scipy.stats import poisson
import telebot
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Model, load_model
from xgboost import XGBClassifier
import urllib3

# Desactivar advertencias de SSL en entornos de producción
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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
# CONFIGURACIÓN DE CREDENCIALES Y SEGURIDAD
# ==========================================
API_KEY = "40f348e5bd5646daa606c0096cb13fb2"
TELEGRAM_TOKEN = "8686243895:AAE5GsHVhBTN9DbEgIimtLa4XtMBkKMs5fw"

bot = telebot.TeleBot(TELEGRAM_TOKEN)

# Reconstrucción dinámica y segura de la URL oficial v4 para evitar recortes de formato
partes_url = ["https://", "api.", "football-data.org", "/v4"]
BASE_URL = "".join(partes_url)

# Filtro estricto al límite real del plan gratuito (Evita llamadas 403 basura)
SEASONS_PAST = [2023, 2024, 2025]
CURRENT_SEASON = 2026

# Sesión segura con Headers completos para evitar bloqueos perimetrales del servidor (Anti-Cloudflare)
session = requests.Session()
session.headers.update({
    "X-Auth-Token": API_KEY,
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Mobile"
})

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
    "BSA": "Campeonato Brasileiro Série A (Brasil)"
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
    """Carga temporadas pasadas desde CSV local y concatena los juegos de la temporada actual."""
    csv_file = f"cache_{league_code}_historical.csv"
    df_past = pd.DataFrame()

    if os.path.exists(csv_file):
        try:
            df_past = pd.read_csv(csv_file)
            df_past["utcDate"] = pd.to_datetime(df_past["utcDate"])
        except Exception as e:
            print(f"Error leyendo CSV local de {league_code}: {e}")

    if df_past.empty:
        past_matches = []
        for s in SEASONS_PAST:
            url = f"{BASE_URL}/competitions/{league_code}/matches?season={s}&status=FINISHED"
            try:
                res = session.get(url, verify=False)
                if res.status_code == 200:
                    data = res.json().get("matches", [])
                    for m in data:
                        past_matches.append({
                            "season": s,
                            "utcDate": m.get("utcDate"),
                            "home": m["homeTeam"]["name"],
                            "away": m["awayTeam"]["name"],
                            "h_g": m["score"]["fullTime"]["home"],
                            "a_g": m["score"]["fullTime"]["away"],
                        })
                    print(f"-> Temporada {s} guardada para {league_code} ({len(data)} partidos)")
                time.sleep(2.0)  # Respetar el límite de velocidad de la API (10 req/min)
            except Exception as e:
                print(f"Error descargando temporada {s} en {league_code}: {e}")

        df_past = pd.DataFrame(past_matches)
        if not df_past.empty:
            df_past.to_csv(csv_file, index=False)
            df_past["utcDate"] = pd.to_datetime(df_past["utcDate"])

    # Descargar la información más fresca disponible de la temporada 2026
    current_matches = []
    url_curr = f"{BASE_URL}/competitions/{league_code}/matches?season={CURRENT_SEASON}&status=FINISHED"
    try:
        res = session.get(url_curr, verify=False)
        if res.status_code == 200:
            data = res.json().get("matches", [])
            for m in data:
                current_matches.append({
                    "season": CURRENT_SEASON,
                    "utcDate": m.get("utcDate"),
                    "home": m["homeTeam"]["name"],
                    "away": m["awayTeam"]["name"],
                    "h_g": m["score"]["fullTime"]["home"],
                    "a_g": m["score"]["fullTime"]["away"],
                })
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
# CÁLCULOS DE CÓRNERS Y POISSON
# ==========================================
def get_team_corners_history_estimated(df, team_name, mean_league_gf, window=5):
    team_matches = df[(df["home"] == team_name) | (df["away"] == team_name)].copy()
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

    ratio_ataque = (avg_gf / max(0.5, mean_league_gf)) if mean_league_gf > 0 else 1.0
    ratio_defensa = (avg_ga / max(0.5, mean_league_gf)) if mean_league_gf > 0 else 1.0

    c_favor_est = max(2.5, min(8.5, 4.75 * (0.6 * ratio_ataque + 0.4)))
    c_contra_est = max(2.5, min(8.5, 4.5 * (0.6 * ratio_defensa + 0.4)))

    return c_favor_est, c_contra_est

def calcular_probabilidades_corners(lambda_corners, lineas=[7.5, 8.5, 9.5, 10.5, 11.5]):
    probs = {}
    for linea in lineas:
        k_max = int(np.floor(linea))
        prob_under = sum(poisson.pmf(k, lambda_corners) for k in range(k_max + 1))
        probs[f"+{linea}"] = (1 - prob_under) * 100
    return probs

def calcular_probabilidades_poisson(lambda_home, lambda_away, max_goles=6):
    prob_matrix = np.zeros((max_goles, max_goles))
    for i in range(max_goles):
        for j in range(max_goles):
            prob_matrix[i, j] = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)

    p_home = np.sum(np.tril(prob_matrix, -1))
    p_draw = np.sum(np.diag(prob_matrix))
    p_away = np.sum(np.triu(prob_matrix, 1))

    total = p_home + p_draw + p_away
    return p_home / total, p_draw / total, p_away / total

# ==========================================
# MOTOR DE SIMULACIÓN MONTE CARLO
# ==========================================
def simular_partido_monte_carlo(lambda_home, lambda_away, lambda_corners, n_simulaciones=10000):
    """
    Ejecuta simulaciones en bloque de manera eficiente usando numpy.
    Mantiene la base del modelo pero expande el espectro de análisis en Telegram.
    """
    # Simulación masiva vectorizada mediante distribución de Poisson
    goles_home_sim = np.random.poisson(lambda_home, n_simulaciones)
    goles_away_sim = np.random.poisson(lambda_away, n_simulaciones)
    corners_sim = np.random.poisson(lambda_corners, n_simulaciones)
    
    # 1. Encontrar el marcador exacto más repetido (Moda)
    marcadores = {}
    for h, a in zip(goles_home_sim, goles_away_sim):
        key = f"{h} - {a}"
        marcadores[key] = marcadores.get(key, 0) + 1
        
    sorted_marcadores = sorted(marcadores.items(), key=lambda x: x[1], reverse=True)
    marcador_top, conteo_top = sorted_marcadores[0]
    prob_marcador_top = (conteo_top / n_simulaciones) * 100
    
    # Obtener alternativas (puestos 2 y 3)
    marcador_alt1 = sorted_marcadores[1][0] if len(sorted_marcadores) > 1 else "0 - 0"
    prob_alt1 = (sorted_marcadores[1][1] / n_simulaciones) * 100 if len(sorted_marcadores) > 1 else 0.0
    
    marcador_alt2 = sorted_marcadores[2][0] if len(sorted_marcadores) > 2 else "1 - 1"
    prob_alt2 = (sorted_marcadores[2][1] / n_simulaciones) * 100 if len(sorted_marcadores) > 2 else 0.0
    
    # 2. Rango de goles más probable (Frecuencia agrupada)
    total_goles_sim = goles_home_sim + goles_away_sim
    rango_2_3 = np.sum((total_goles_sim >= 2) & (total_goles_sim <= 3))
    prob_rango_goles = (rango_2_3 / n_simulaciones) * 100
    
    # 3. Margen de volatilidad de córners (Desviación estándar empírica)
    media_c = np.mean(corners_sim)
    std_c = np.std(corners_sim)
    lim_inf = max(0, int(np.floor(media_c - std_c)))
    lim_sup = int(np.ceil(media_c + std_c))
    
    rango_c = np.sum((corners_sim >= lim_inf) & (corners_sim <= lim_sup))
    prob_rango_c = (rango_c / n_simulaciones) * 100
    
    return {
        "marcador_top": marcador_top,
        "prob_marcador_top": prob_marcador_top,
        "marcador_alt1": marcador_alt1,
        "prob_alt1": prob_alt1,
        "marcador_alt2": marcador_alt2,
        "prob_alt2": prob_alt2,
        "prob_rango_goles": prob_rango_goles,
        "lim_inf_c": lim_inf,
        "lim_sup_c": lim_sup,
        "prob_rango_c": prob_rango_c
    }

# ==========================================
# PROCESAMIENTO DE HISTORIAL
# ==========================================
def get_team_history(df, team_name, window=5):
    team_matches = df[(df["home"] == team_name) | (df["away"] == team_name)].copy()
    
    if len(team_matches) < window:
        global_mean_h = df["h_g"].mean() if len(df) > 0 else 1.4
        global_mean_a = df["a_g"].mean() if len(df) > 0 else 1.1
        return np.tile([global_mean_h, global_mean_a], (window, 1))

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
# GENERACIÓN DE REPORTE Y OBTENCIÓN IA
# ==========================================
def verificar_antiguedad_modelo(ruta_archivo, dias_limite=7):
    """Retorna True si el archivo no existe o si es más viejo que los días límite."""
    if not os.path.exists(ruta_archivo):
        return True
    tiempo_creacion = os.path.getmtime(ruta_archivo)
    fecha_creacion = datetime.fromtimestamp(tiempo_creacion)
    if datetime.now() - fecha_creacion > timedelta(days=dias_limite):
        return True
    return False

def ejecutar_modelo_y_generar_reporte(league_code, dias_a_futuro=7):
    df = get_historical_data_cached(league_code)
    if df.empty or "home" not in df.columns:
        return f"❌ No se pudieron extraer datos históricos para *{league_code}*."

    X_s, X_st, y_train = prepare_engine(df)
    if len(X_s) == 0:
        return f"⚠️ No hay suficientes datos procesables para *{league_code}*."

    keras_model_path = f"modelo_lstm_{league_code}.keras"
    xgb_model_path = f"modelo_xgb_{league_code}.pkl"

    # Forzar reentrenamiento automático si el modelo supera el tiempo límite
    forzar_reentrenamiento = verificar_antiguedad_modelo(keras_model_path, dias_limite=7)
    if forzar_reentrenamiento:
        print(f"🔄 Reentrenando modelos para {league_code} por obsolescencia (Límite: 7 días)...")
        if os.path.exists(keras_model_path): os.remove(keras_model_path)
        if os.path.exists(xgb_model_path): os.remove(xgb_model_path)

    # 1. Cargar o Entrenar Modelo LSTM
    if os.path.exists(keras_model_path):
        model = load_model(keras_model_path)
        extractor = Model(inputs=model.inputs, outputs=model.layers[2].output)
    else:
        inp = Input(shape=(5, 4))
        x = LSTM(32, return_sequences=False)(inp)
        x = Dropout(0.2)(x)
        fe = Dense(16, activation="relu")(x)
        out = Dense(3, activation="softmax")(fe)

        model = Model(inputs=inp, outputs=out)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(X_s, y_train, epochs=15, batch_size=16, verbose=0)
        model.save(keras_model_path)
        extractor = Model(inputs=inp, outputs=fe)

    # 2. Extractor + XGBoost Classifier
    extracted_features = extractor.predict(X_s, verbose=0)
    X_final = np.hstack([extracted_features, X_st])

    if os.path.exists(xgb_model_path):
        xgb = joblib.load(xgb_model_path)
    else:
        xgb = XGBClassifier(
            n_estimators=100, learning_rate=0.03, max_depth=3,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        ).fit(X_final, y_train)
        joblib.dump(xgb, xgb_model_path)

    # 3. Consulta de Próximos Partidos
    url = f"{BASE_URL}/competitions/{league_code}/matches?status=SCHEDULED,IN_PLAY,PAUSED"
    res = session.get(url, verify=False)
    if res.status_code != 200:
        return f"⚠️ Error consultando partidos ({res.status_code})."

    proximos = res.json().get("matches", [])
    if not proximos:
        nombre_liga = LIGAS_DISPONIBLES.get(league_code, league_code)
        return f"⚠️ No hay encuentros programados para *{nombre_liga}*."

    ahora = datetime.now(timezone.utc)
    limite_fecha = ahora + timedelta(days=dias_a_futuro)

    partidos_filtrados = []
    for m in proximos:
        fecha_match = datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00"))
        if m.get("status") in ["IN_PLAY", "PAUSED"] or (ahora <= fecha_match <= limite_fecha):
            partidos_filtrados.append(m)

    if not partidos_filtrados:
        proximos = sorted(proximos, key=lambda x: x["utcDate"])[:10]
    else:
        partidos_filtrados = sorted(partidos_filtrados, key=lambda x: x["utcDate"])
        proximos = partidos_filtrados

    # 4. Formatear reporte analítico final con Monte Carlo integrado
    nombre_liga = LIGAS_DISPONIBLES.get(league_code, league_code)
    reporte = f"📊 *REPORTE ANALÍTICO PRO: {nombre_liga}*\n"
    reporte += "═" * 30 + "\n\n"

    mean_h_g = df["h_g"].mean() if len(df) > 0 else 1.4
    mean_a_g = df["a_g"].mean() if len(df) > 0 else 1.1

    for m in proximos:
        h_name, a_name = m["homeTeam"]["name"], m["awayTeam"]["name"]
        fecha = m["utcDate"][:10]
        jornada = m.get("matchday", "N/A")
        estado_str = "🔴 EN VIVO" if m.get("status") in ["IN_PLAY", "PAUSED"] else f"📅 JORNADA {jornada}"

        r_h = get_team_history(df, h_name, window=5)
        r_a = get_team_history(df, a_name, window=5)

        h_gf, h_ga = np.mean(r_h[:, 0]), np.mean(r_h[:, 1])
        a_gf, a_ga = np.mean(r_a[:, 0]), np.mean(r_a[:, 1])

        # Cálculo matemático estático base (Brain xG de Poisson)
        avg_h_xg = max(0.2, (h_gf / mean_h_g) * (a_ga / mean_a_g) * mean_h_g)
        avg_a_xg = max(0.2, (a_gf / mean_a_g) * (h_ga / mean_h_g) * mean_a_g)
        total_xg = avg_h_xg + avg_a_xg

        seq = np.hstack([r_h, r_a]).reshape(1, 5, 4)
        feat = extractor.predict(seq, verbose=0)
        static_feats = np.array([[h_gf, h_ga, a_gf, a_ga, mean_h_g, mean_a_g]])
        
        prob_ia = xgb.predict_proba(np.hstack([feat, static_feats]))[0]

        p_h_stat, p_e_stat, p_v_stat = calcular_probabilidades_poisson(avg_h_xg, avg_a_xg)

        p_h = 0.6 * prob_ia[0] + 0.4 * p_h_stat
        p_e = 0.6 * prob_ia[1] + 0.4 * p_e_stat
        p_v = 0.6 * prob_ia[2] + 0.4 * p_v_stat

        total_p = p_h + p_e + p_v
        p_h, p_e, p_v = p_h / total_p, p_e / total_p, p_v / total_p

        p_btts = (1 - poisson.pmf(0, avg_h_xg)) * (1 - poisson.pmf(0, avg_a_xg)) * 100
        p_over25 = (1 - sum(poisson.pmf(i, total_xg) for i in range(3))) * 100

        sugerencia = ""
        if p_h >= 0.55: sugerencia = f"💡 *RECOMENDACIÓN:* Victoria {h_name} ({p_h:.1%})"
        elif p_v >= 0.55: sugerencia = f"💡 *RECOMENDACIÓN:* Victoria {a_name} ({p_v:.1%})"
        elif p_over25 >= 65.0: sugerencia = f"💡 *RECOMENDACIÓN:* Alta probabilidad Over 2.5 ({p_over25:.1f}%)"

        # Análisis dinámico de córners base
        c_h_favor, c_h_contra = get_team_corners_history_estimated(df, h_name, mean_h_g)
        c_a_favor, c_a_contra = get_team_corners_history_estimated(df, a_name, mean_a_g)

        exp_c_home = (c_h_favor + c_a_contra) / 2
        exp_c_away = (c_a_favor + c_h_contra) / 2
        total_exp_corners = exp_c_home + exp_c_away
        probs_corners = calcular_probabilidades_corners(total_exp_corners)

        # DISPARADOR DEL MOTOR MONTE CARLO (10,000 Simulaciones en bloque)
        mc = simular_partido_monte_carlo(avg_h_xg, avg_a_xg, total_exp_corners, n_simulaciones=10000)

        # Ensamblar estructura limpia del mensaje de Telegram
        reporte += f"{estado_str} | 📅 {fecha}\n"
        reporte += f"⚽ *{h_name} vs {a_name}*\n"
        reporte += f"🏆 *Probabilidades IA (1X2):* Local: {p_h:.1%} | Empate: {p_e:.1%} | Visita: {p_v:.1%}\n\n"
        
        reporte += f"🎲 *SIMULACIÓN MONTE CARLO (10,000 partidos):*\n"
        reporte += f"    • *Marcador Top:* {mc['marcador_top']} ({mc['prob_marcador_top']:.1f}%)\n"
        reporte += f"    • *Alternativos:* {mc['marcador_alt1']} ({mc['prob_alt1']:.1f}%) | {mc['marcador_alt2']} ({mc['prob_alt2']:.1f}%)\n"
        reporte += f"    • *Rango de Goles:* 2 a 3 goles ({mc['prob_rango_goles']:.1f}% de prob.)\n\n"
        
        reporte += f"🥅 Goles Est.: {h_name}: {avg_h_xg:.2f} | {a_name}: {avg_a_xg:.2f}\n"
        reporte += f"🔥 Both Score: {p_btts:.1f}% | Over 2.5 Goles: {p_over25:.1f}%\n\n"
        
        if sugerencia: 
            reporte += f"{sugerencia}\n"
            
        reporte += f"🏁 *PROYECCIÓN DE CÓRNERS (Monte Carlo):*\n"
        reporte += f"    • *Córners Estimados:* {h_name}: {exp_c_home:.1f} | {a_name}: {exp_c_away:.1f} (Total: {total_exp_corners:.1f})\n"
        reporte += f"    • *Volatilidad:* Entre {mc['lim_inf_c']} y {mc['lim_sup_c']} córners ({mc['prob_rango_c']:.1f}% de los escenarios)\n\n"
        
        reporte += f"📈 *Probabilidades Córners:*\n"
        reporte += f"    • +7.5: {probs_corners['+7.5']:.1f}% | +8.5: {probs_corners['+8.5']:.1f}%\n"
        reporte += f"    • +9.5: {probs_corners['+9.5']:.1f}% | +10.5: {probs_corners['+10.5']:.1f}%\n"
        reporte += f"    • +11.5: {probs_corners['+11.5']:.1f}%\n"
        reporte += "─────────────────────────────\n"

    return reporte

# ==========================================
# HANDLERS DEL BOT DE TELEGRAM
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
    bot.reply_to(message, bienvenida + obtener_menu_ligas(), parse_mode="Markdown")

@bot.message_handler(func=lambda msg: True)
def responder_prompt(message):
    texto = message.text.strip().upper()

    if texto in LIGAS_DISPONIBLES:
        nombre_liga = LIGAS_DISPONIBLES[texto]
        bot.reply_to(message, f"⏳ Procesando datos e impulsando modelos IA para *{nombre_liga}* ({texto})... por favor espera.", parse_mode="Markdown")
        try:
            resultado = ejecutar_modelo_y_generar_reporte(texto, dias_a_futuro=7)
            enviar_mensaje_largo(message.chat.id, resultado)
        except Exception as e:
            bot.reply_to(message, f"❌ Ocurrió un error al procesar la solicitud: {e}")
    else:
        bot.reply_to(message, f"⚠️ Código de liga no reconocido.\n\n" + obtener_menu_ligas(), parse_mode="Markdown")

if __name__ == "__main__":
    # Arrancar servidor Flask en un hilo independiente para el Keep-Alive de Render
    Thread(target=run_flask).start()

    print("🤖 Bot de Telegram activo y esperando comandos...")
    bot.infinity_polling(timeout=10, long_polling_timeout=5)
