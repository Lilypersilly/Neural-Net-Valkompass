import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os
import json
from datetime import datetime
from neural_net import (
    load_party_datasets,
    load_example_essays,
    load_neural_engine,
    smart_route_sentences,
    evaluate_routed_profiles,
    calculate_absolute_podium,
    generate_execution_log
)

st.set_page_config(
    page_title="Neural Political Compass",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

datasets, party_colors, party_metadata = load_party_datasets("datasets")
example_profiles = load_example_essays("exempel")
model_package, device = load_neural_engine()
dataset_keys = list(datasets.keys())

st.sidebar.title("Systemstatus")
st.sidebar.info(f"Kör på enhet: **{device.upper()}**")

st.sidebar.markdown("---")
st.sidebar.subheader(f"Inlästa Partier ({len(party_colors)})")

if not party_colors:
    st.sidebar.warning("Inga partier hittades i mappen /datasets.")
else:
    for party, color in party_colors.items():
        st.sidebar.markdown(
            f"<div style='margin-bottom: 6px;'>"
            f"<span style='display:inline-block; width:12px; height:12px; background-color:{color}; border-radius:50%; margin-right:8px;'></span>"
            f"<strong>{party}</strong>"
            f"</div>",
            unsafe_allow_html=True
        )

st.sidebar.markdown("---")
if st.sidebar.button("Läs om dataset & exempel"):
    st.rerun()

st.title("Neural Political Compass")
st.markdown(
    "Analysera åsikter och retoriska formuleringar mot de politiska partiernas ståndpunkter "
    "med en tvåstegs SBERT + mDeBERTa NLI neural inferensmotor."
)

st.markdown("---")

st.subheader("1. Beskriv dina politiska ståndpunkter")

mode_selection = st.radio(
    "Välj inmatningsmetod:",
    ["Egen text (Flikar per ämnesområde)", "Ladda fördefinierat testexempel från /exempel"],
    horizontal=True
)

user_inputs = {}

if mode_selection == "Ladda fördefinierat testexempel från /exempel":
    if not example_profiles:
        st.warning("Inga exempel-JSON-filer hittades i mappen /exempel.")
        example_choices = {"Välj ett exempel...": {"Economy": "", "Climate": "", "Immigration": "", "Law And Order": ""}}
    else:
        example_choices = {"Välj ett exempel...": {"Economy": "", "Climate": "", "Immigration": "", "Law And Order": ""}}
        example_choices.update(example_profiles)

    selected_example = st.selectbox("Välj ett testscenario:", list(example_choices.keys()))
    chosen_profile = example_choices[selected_example]
    
    user_inputs = chosen_profile.copy()
    
    col_prev1, col_prev2 = st.columns(2)
    with col_prev1:
        st.text_area("Ekonomi (förhandsvisning):", value=user_inputs.get("Economy", ""), height=90, disabled=True)
        st.text_area("Klimat (förhandsvisning):", value=user_inputs.get("Climate", ""), height=90, disabled=True)
    with col_prev2:
        st.text_area("Immigration (förhandsvisning):", value=user_inputs.get("Immigration", ""), height=90, disabled=True)
        st.text_area("Lag & Rätt (förhandsvisning):", value=user_inputs.get("Law And Order", ""), height=90, disabled=True)
        
    has_input = (selected_example != "Välj ett exempel...") and any(txt.strip() for txt in user_inputs.values())
else:
    tabs = st.tabs(["Ekonomi & Skatter", "Klimat & Miljö", "Migration & Asyl", "Lag & Rätt"])
    
    with tabs[0]:
        user_inputs["Economy"] = st.text_area(
            "Ekonomisk politik, skatter och välfärd:",
            placeholder="Skriv dina tankar om vinster i välfärden, skatt på arbete vs kapital, bidrag...",
            height=140
        )
    with tabs[1]:
        user_inputs["Climate"] = st.text_area(
            "Klimat, energi och miljö:",
            placeholder="Skriv dina tankar om kärnkraft, förnybar energi, bränsleskatter, skogsbruk...",
            height=140
        )
    with tabs[2]:
        user_inputs["Immigration"] = st.text_area(
            "Migrationspolitik och integration:",
            placeholder="Skriv dina tankar om asylrätt, medborgarskapskrav, återvandring...",
            height=140
        )
    with tabs[3]:
        user_inputs["Law And Order"] = st.text_area(
            "Kriminalpolitik och rättsväsende:",
            placeholder="Skriv dina tankar om straffsatser, polisresurser, drogpolitik...",
            height=140
        )
    has_input = any(txt.strip() for txt in user_inputs.values())

if has_input:
    col_save1, col_save2 = st.columns([2, 3])
    with col_save1:
        if st.button("Spara nuvarande text som json i /exempel"):
            os.makedirs("exempel", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"exempel_{timestamp}.json"
            filepath = os.path.join("exempel", filename)
            
            payload = {
                "title": f"Nytt exempel {timestamp}",
                "pillars": user_inputs
            }
            
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=4)
                st.success(f"Sparad till {filepath}! Byt namn på filen manuellt vid behov.")
            except Exception as e:
                st.error(f"Kunde inte spara filen: {e}")

st.markdown("<br>", unsafe_allow_html=True)



if st.button("Utför Retorisk & Logisk Analys", type="primary", disabled=not has_input):
    if not party_colors:
        st.error("Inga partier inlästa. Lägg till JSON-filer i mappen datasets.")
        st.stop()
        
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    
    def update_progress(current, total, pillar, party):
        pct = min(1.0, current / max(1, total))
        progress_bar.progress(pct)
        status_text.text(f"Inferens pågår: Utvärderar {pillar} för {party}... ({int(pct*100)}%)")

    status_text.text("Steg 1/2: Genomför semantisk ruttning...")
    routed = smart_route_sentences(user_inputs, dataset_keys, model_package)

    status_text.text("Steg 2/2: Genomför logisk inferens per sats...")
    
    # Kör inferensen (räknar ut alla 3 metoder automatiskt)
    pillar_scores_all_modes, inference_traces = evaluate_routed_profiles(
        routed, 
        datasets, 
        model_package, 
        progress_callback=update_progress
    )
    
    progress_bar.progress(1.0)
    status_text.success("Analys slutförd!")
    
    st.session_state["inference_traces"] = inference_traces
    st.session_state["pillar_scores_all"] = pillar_scores_all_modes
    st.session_state["routed"] = routed

# --- RESULTAT & DYNAMISK UPPDATERING ---
if "pillar_scores_all" in st.session_state:
    st.markdown("---")
    
    col_titel, col_radio = st.columns([1, 1])
    with col_titel:
        st.subheader("2. Resultat & Matchningspodium")
    with col_radio:
        scaling_mode_selection = st.radio(
            "Dataskalning (ändrar hur datan skalas mening för mening):",
            ["Standard", "SANN", "Relativ"],
            horizontal=True
        )
    
    # Tydlig förklaring direkt under valen
    mode_descriptions = {
        "Standard": " *Standard (Rekommenderad): Rensar bort brus och ger tydliga, balanserade valkompass-resultat.*",
        "SANN": " *SANN: Visar rå cosinus likhet utan trösklar. Gynnar partier med bred och allmän retorik.*",
        "Relativ": " *Relativ: Analysläge som ställer partierna mot varandra per mening för maxad kontrast.*"
    }
    st.caption(mode_descriptions.get(scaling_mode_selection, ""))

    mode = scaling_mode_selection
    
    # Hämta den data som motsvarar användarens val
    pillar_scores = st.session_state["pillar_scores_all"][mode]
    inference_traces = st.session_state["inference_traces"]
    routed = st.session_state["routed"]
    
    podium = calculate_absolute_podium(pillar_scores)
    top_parties = list(podium.items())
    
    cols = st.columns(min(3, len(top_parties)))
    ranks = ["1:a Plats", "2:a Plats", "3:e Plats"]
    for idx in range(min(3, len(top_parties))):
        p_name, p_score = top_parties[idx]
        p_color = party_colors.get(p_name, "#888888")
        with cols[idx]:
            st.markdown(
                f"""
                <div style="border-top: 5px solid {p_color}; background-color: rgba(255,255,255,0.04); padding: 16px; border-radius: 8px;">
                    <h4 style="margin:0; color: #BBB;">{ranks[idx]}</h4>
                    <h2 style="margin:4px 0 10px 0; color: {p_color};">{p_name}</h2>
                    <h3 style="margin:0;">{p_score*100:.1f}% Match</h3>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # BAR CHART & RADAR
    col_bar, col_radar = st.columns([1.1, 0.9])
    with col_bar:
        st.markdown(f"#### Total Matchning ({mode})")
        df_podium = pd.DataFrame([{"Parti": k, "Matchning": v * 100, "Färg": party_colors.get(k, "#888888")} for k, v in podium.items()])
        fig_bar = px.bar(df_podium, x="Matchning", y="Parti", orientation="h", text=df_podium["Matchning"].apply(lambda x: f"{x:.1f}%"), color="Parti", color_discrete_map=party_colors)
        fig_bar.update_layout(showlegend=False, xaxis=dict(range=[0, 100]), yaxis=dict(autorange="reversed"), height=380, margin=dict(l=10, r=20, t=10, b=10))
        st.plotly_chart(fig_bar, use_container_width=True, key="podium_bar_chart")

    with col_radar:
        st.markdown("#### Matchningsprofil över Sakområden")
        radar_pillars = [p for p in dataset_keys if p in pillar_scores and pillar_scores[p]]
        if radar_pillars:
            fig_radar = go.Figure()
            for p_name, _ in top_parties[:4]:
                scores = [pillar_scores[p].get(p_name, 0.0) * 100 for p in radar_pillars]
                scores.append(scores[0])
                fig_radar.add_trace(go.Scatterpolar(r=scores, theta=radar_pillars + [radar_pillars[0]], name=p_name, line=dict(color=party_colors.get(p_name, "#888888"), width=2), fill='toself', opacity=0.25))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, height=380, margin=dict(l=40, r=40, t=20, b=20))
            st.plotly_chart(fig_radar, use_container_width=True, key="podium_radar_chart")

    # DETALJER PER SAKOMRÅDE
    st.markdown("---")
    st.subheader("3. Detaljerad Uppdelning per Sakområde")
    pillar_tabs = st.tabs([f"{p}" for p in dataset_keys])
    for idx, pillar in enumerate(dataset_keys):
        with pillar_tabs[idx]:
            if pillar in pillar_scores and pillar_scores[pillar]:
                p_sorted = sorted(pillar_scores[pillar].items(), key=lambda x: x[1], reverse=True)
                c1, c2 = st.columns([1, 1])
                with c1:
                    df_pillar = pd.DataFrame([{"Parti": k, "Poäng": v * 100} for k, v in p_sorted])
                    fig_pbar = px.bar(df_pillar, x="Poäng", y="Parti", orientation="h", text=df_pillar["Poäng"].apply(lambda x: f"{x:.1f}%"), color="Parti", color_discrete_map=party_colors)
                    fig_pbar.update_layout(showlegend=False, xaxis=dict(range=[0, 100]), yaxis=dict(autorange="reversed"), height=300, margin=dict(l=10, r=20, t=10, b=10))
                    st.plotly_chart(fig_pbar, use_container_width=True, key=f"pillar_bar_{pillar}")
                with c2:
                    st.markdown("**Analyserade textstycken:**")
                    for c in routed.get(pillar, []):
                        st.info(f"”{c}”")
            else:
                st.write("Ingen data.")

    # EXPLAINER
    st.markdown("---")
    st.subheader("4. Neural Rationale & Förklaring")
    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        selected_explain_party = st.selectbox("Välj parti att granska:", list(party_colors.keys()))
    with exp_col2:
        selected_explain_pillar = st.selectbox("Välj sakområde:", dataset_keys)

    if selected_explain_party in inference_traces.get(selected_explain_pillar, {}):
        traces = inference_traces[selected_explain_pillar][selected_explain_party]
        if traces:
            for idx, t in enumerate(traces):
                if "scores" not in t or t["scores"][mode] == 0.0:
                    continue
                
                display_score = t["scores"][mode]
                status_color = "#2ECC71" if "Stöd" in t["nli_status"] else ("#E74C3C" if "Konflikt" in t["nli_status"] else "#F39C12")
                statement_display = f"<strong>{t.get('topic', '')}:</strong> {t.get('stance', '')}"
                
                st.markdown(
                    f"""
                    <div style="background-color: rgba(255,255,255,0.03); border-left: 4px solid {status_color}; padding: 12px; border-radius: 6px; margin-bottom: 6px;">
                        <strong style="color: #DDD;">Din mening:</strong> <span style="color: #FFF;">"{t['user_chunk']}"</span><br>
                        <strong style="color: #BBB;">Partiståndpunkt ({selected_explain_party}):</strong> <span style="color: #EEE;">{statement_display}</span><br>
                        <hr style="margin: 6px 0; border-color: #333;">
                        <span style="color: {status_color}; font-weight: bold;">{t['nli_status']}</span> | 
                        <span style="color: #888;">Matchningspoäng ({mode}): <strong>{display_score:.1f}%</strong></span>
                    </div>
                    """, unsafe_allow_html=True
                )

                # --- FEEDBACK / KORRIGERING FÖR NLI DATASET ---
                feedback_expander_key = f"exp_{selected_explain_pillar}_{selected_explain_party}_{idx}"
                with st.expander("Håller du inte med inferensen? (Klicka för att korrigera)", expanded=False):
                    col_fb1, col_fb2 = st.columns([2, 1])
                    with col_fb1:
                        user_correction = st.radio(
                            "Vad är den faktiska logiska relationen?",
                            ["Medhäll / Stöd (Entailment)", "Neutral / Delvis", "Konflikt / Motsägelse (Contradiction)"],
                            key=f"radio_{feedback_expander_key}"
                        )
                    with col_fb2:
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button("Spara till NLI-dataset", key=f"btn_{feedback_expander_key}"):
                            if "Medhäll" in user_correction:
                                corrected_label = "entailment"
                            elif "Neutral" in user_correction:
                                corrected_label = "neutral"
                            else:
                                corrected_label = "contradiction"

                            dataset_path = "nli_training_dataset.json"
                            existing_data = []
                            if os.path.exists(dataset_path):
                                try:
                                    with open(dataset_path, "r", encoding="utf-8") as f:
                                        existing_data = json.load(f)
                                except Exception:
                                    existing_data = []

                            new_entry = {
                                "premise": t["user_chunk"],
                                "hypothesis": t["stance"] if t.get("stance") else t.get("party_statement", ""),
                                "label": corrected_label,
                                "party": selected_explain_party,
                                "pillar": selected_explain_pillar,
                                "original_model_status": t["nli_status"]
                            }

                            existing_data.append(new_entry)

                            try:
                                with open(dataset_path, "w", encoding="utf-8") as f:
                                    json.dump(existing_data, f, ensure_ascii=False, indent=4)
                                st.success(f"Sparat till {dataset_path}! (Totalt {len(existing_data)} ex)")
                            except Exception as e:
                                st.error(f"Kunde inte spara: {e}")
        else:
            st.info(f"Inga aktiva textstycken matchades för {selected_explain_party} inom {selected_explain_pillar}.")
    else:
        st.info("Ingen data tillgänglig för denna kombination.")