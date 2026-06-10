# -*- coding: utf-8 -*-
"""
EkonoMiner — Yapay Zekâ Destekli Küresel Ekonomik Veri Madenciliği Platformu
İKT-442 Veri Madenciliği Final Projesi
Geliştiren: Affan Aybars Damgacı (230502050175)

Veri Kaynağı : Dünya Bankası (World Bank Open Data API) + yerel yedek veri seti
Veri Madenciliği : K-Means kümeleme, PCA boyut indirgeme, korelasyon analizi, anomali tespiti
Yapay Zekâ : Google Gemini API (otomatik ekonomik yorum / analist raporu)
Entegrasyon : Google Sheets (analiz sonuçlarının buluta kaydı)
"""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
# Sayfa ayarları
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="EkonoMiner | Ekonomik Veri Madenciliği",
    page_icon="🌍",
    layout="wide",
)

GOSTERGELER = {
    "kisi_basi_gsyh": ("Kişi Başı GSYH (USD)", "NY.GDP.PCAP.CD"),
    "gsyh_buyume": ("GSYH Büyümesi (%)", "NY.GDP.MKTP.KD.ZG"),
    "enflasyon": ("Enflasyon (%)", "FP.CPI.TOTL.ZG"),
    "issizlik": ("İşsizlik (%)", "SL.UEM.TOTL.ZS"),
    "ihracat_gsyh": ("İhracat / GSYH (%)", "NE.EXP.GNFS.ZS"),
    "yasam_beklentisi": ("Yaşam Beklentisi (yıl)", "SP.DYN.LE00.IN"),
    "internet_kullanimi": ("İnternet Kullanımı (%)", "IT.NET.USER.ZS"),
    "kentlesme": ("Kentleşme Oranı (%)", "SP.URB.TOTL.IN.ZS"),
}
ETIKET = {k: v[0] for k, v in GOSTERGELER.items()}


# ----------------------------------------------------------------------------
# Veri katmanı
# ----------------------------------------------------------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def yerel_veri_yukle() -> pd.DataFrame:
    yol = os.path.join(os.path.dirname(__file__), "data", "worldbank_snapshot.csv")
    return pd.read_csv(yol)


@st.cache_data(ttl=3600, show_spinner=False)
def worldbank_canli_cek(iso_listesi: tuple, yil: str = "2023") -> pd.DataFrame:
    """Dünya Bankası API'den seçili ülkeler için tüm göstergeleri çeker.
    Hata durumunda exception fırlatır (böylece başarısız sonuç önbelleğe alınmaz)."""
    ulkeler = ";".join(iso_listesi)
    kayitlar = {}
    basarili_gosterge = 0
    for kolon, (_, kod) in GOSTERGELER.items():
        url = (
            f"https://api.worldbank.org/v2/country/{ulkeler}/indicator/{kod}"
            f"?format=json&date={yil}&per_page=400"
        )
        # Her gösterge için 2 deneme; tek göstergenin hatası tüm çekimi bozmasın
        for deneme in range(2):
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                yanit = r.json()
                if len(yanit) < 2 or yanit[1] is None:
                    break
                for satir in yanit[1]:
                    iso = satir["countryiso3code"]
                    kayitlar.setdefault(iso, {})[kolon] = satir["value"]
                basarili_gosterge += 1
                break
            except Exception:
                continue
    if basarili_gosterge < 5 or len(kayitlar) < 20:
        raise RuntimeError("World Bank API'den yeterli veri alınamadı")
    df = pd.DataFrame.from_dict(kayitlar, orient="index").reset_index(names="iso3")
    # Türkçe ülke adlarını yerel sözlükten eşle
    adlar = yerel_veri_yukle().set_index("iso3")["ulke"].to_dict()
    df["ulke"] = df["iso3"].map(adlar).fillna(df["iso3"])
    return df.dropna(thresh=6)


def veri_getir(kaynak: str) -> tuple[pd.DataFrame, str]:
    yerel = yerel_veri_yukle()
    if kaynak == "Dünya Bankası API (canlı)":
        try:
            with st.spinner("Dünya Bankası API'den veri çekiliyor..."):
                canli = worldbank_canli_cek(tuple(yerel["iso3"]))
            if len(canli) >= 20:
                return canli, "🌐 Canlı veri: Dünya Bankası Open Data API (2023)"
        except Exception:
            pass
        st.sidebar.warning("API'ye ulaşılamadı, yerel yedek veri kullanılıyor. (Tekrar denemek için kaynağı değiştirip geri alın)")
    return yerel, "💾 Yerel veri: Dünya Bankası 2023 anlık görüntüsü (yedek)"


# ----------------------------------------------------------------------------
# Yapay zekâ katmanı (Google Gemini)
# ----------------------------------------------------------------------------
def gemini_anahtari() -> str | None:
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.environ.get("GEMINI_API_KEY")


def gemini_sor(istem: str) -> tuple[str | None, str]:
    """Gemini REST API ile metin üretir. (yanıt, hata_detayı) döner."""
    anahtar = gemini_anahtari()
    if not anahtar:
        return None, "API anahtarı tanımlı değil (Secrets içine GEMINI_API_KEY ekleyin)."
    hatalar = []
    modeller = (
        "gemini-flash-latest",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
    )
    for model in modeller:
        try:
            url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model}:generateContent?key={anahtar}"
            )
            govde = {"contents": [{"parts": [{"text": istem}]}]}
            r = requests.post(url, json=govde, timeout=60)
            if r.status_code != 200:
                try:
                    mesaj = r.json().get("error", {}).get("message", "")[:120]
                except Exception:
                    mesaj = r.text[:120]
                hatalar.append(f"{model} → HTTP {r.status_code}: {mesaj}")
                continue
            return r.json()["candidates"][0]["content"]["parts"][0]["text"], ""
        except Exception as e:
            hatalar.append(f"{model} → {type(e).__name__}: {str(e)[:100]}")
            continue
    return None, " | ".join(hatalar)


def yerlesik_yorum(df: pd.DataFrame, kume_ozet: pd.DataFrame) -> str:
    """API anahtarı yoksa devreye giren kural tabanlı yedek yorumlama motoru."""
    metin = ["**Otomatik Analiz Özeti (yerleşik motor):**\n"]
    for kume_no, satir in kume_ozet.iterrows():
        uyeler = df[df["Küme"] == kume_no]["ulke"].tolist()
        profil = []
        if satir["kisi_basi_gsyh"] > df["kisi_basi_gsyh"].median():
            profil.append("yüksek gelirli")
        else:
            profil.append("düşük/orta gelirli")
        if satir["enflasyon"] > df["enflasyon"].median():
            profil.append("enflasyonist baskı altında")
        if satir["gsyh_buyume"] > df["gsyh_buyume"].median():
            profil.append("güçlü büyüyen")
        metin.append(
            f"- **Küme {kume_no}** ({len(uyeler)} ülke): {', '.join(profil)} ekonomiler. "
            f"Örnek üyeler: {', '.join(uyeler[:6])}."
        )
    return "\n".join(metin)


# ----------------------------------------------------------------------------
# Google Sheets entegrasyonu
# ----------------------------------------------------------------------------
def sheets_kaydet(df: pd.DataFrame, sayfa_adi: str) -> tuple[bool, str]:
    """st.secrets içindeki servis hesabıyla sonuçları Google Sheets'e yazar."""
    try:
        import gspread
        from google.oauth2.service_account import Credentials

        bilgiler = dict(st.secrets["gcp_service_account"])
        kapsam = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        kimlik = Credentials.from_service_account_info(bilgiler, scopes=kapsam)
        istemci = gspread.authorize(kimlik)
        dosya = istemci.open_by_key(st.secrets["GSHEET_ID"])
        try:
            ws = dosya.worksheet(sayfa_adi)
            dosya.del_worksheet(ws)
        except Exception:
            pass
        ws = dosya.add_worksheet(title=sayfa_adi, rows=len(df) + 5, cols=len(df.columns) + 2)
        # NaN/inf değerleri Google'ın kabul etmesi için boş hücreye çevir
        temiz = df.replace([np.inf, -np.inf], np.nan)
        temiz = temiz.astype(object).where(pd.notna(temiz), "")
        veriler = [temiz.columns.tolist()] + temiz.astype(str).replace("nan", "").values.tolist()
        ws.update(values=veriler)
        return True, f"✅ Sonuçlar Google Sheets'e yazıldı (sayfa: {sayfa_adi})."
    except Exception as e:
        return False, f"Google Sheets bağlantısı kurulamadı: {e}"


# ----------------------------------------------------------------------------
# Arayüz
# ----------------------------------------------------------------------------
st.title("🌍 EkonoMiner")
st.caption(
    "Yapay Zekâ Destekli Küresel Ekonomik Veri Madenciliği Platformu — "
    "İKT-442 Veri Madenciliği Final Projesi | Affan Aybars Damgacı"
)

with st.sidebar:
    st.header("⚙️ Ayarlar")
    kaynak = st.radio(
        "Veri kaynağı",
        ["Dünya Bankası API (canlı)", "Yerel yedek veri (hızlı)"],
        index=1,
        help="Canlı seçenek World Bank Open Data API'den 2023 verilerini çeker.",
    )
    df, kaynak_notu = veri_getir(kaynak)
    st.info(kaynak_notu)

    secili_gostergeler = st.multiselect(
        "Analizde kullanılacak göstergeler",
        options=list(ETIKET.keys()),
        default=list(ETIKET.keys()),
        format_func=lambda k: ETIKET[k],
    )
    k = st.slider("Küme sayısı (K-Means)", 2, 8, 4)
    st.divider()
    st.markdown(
        "🤖 **AI durumu:** "
        + ("Gemini bağlı ✅" if gemini_anahtari() else "Anahtar yok — yerleşik motor devrede")
    )

if len(secili_gostergeler) < 2:
    st.error("Lütfen en az 2 gösterge seçin.")
    st.stop()

# Eksik değerleri medyanla doldur (veri ön işleme)
X_ham = df[secili_gostergeler].copy()
X_ham = X_ham.fillna(X_ham.median(numeric_only=True))
olcekleyici = StandardScaler()
X = olcekleyici.fit_transform(X_ham)

# K-Means + PCA + Anomali tespiti
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["Küme"] = kmeans.fit_predict(X)
pca = PCA(n_components=2, random_state=42)
bilesenler = pca.fit_transform(X)
df["PC1"], df["PC2"] = bilesenler[:, 0], bilesenler[:, 1]
df["Anomali"] = IsolationForest(random_state=42, contamination=0.08).fit_predict(X)

sekme1, sekme2, sekme3, sekme4, sekme5, sekme6 = st.tabs(
    ["📊 Veri Keşfi", "🔗 Korelasyon", "🧩 Kümeleme", "🆚 Ülke Karşılaştırma", "🤖 AI Analist", "📤 Google Sheets"]
)

# --- 1) Veri keşfi -----------------------------------------------------------
with sekme1:
    st.subheader("Veri Seti Keşfi")
    c1, c2, c3 = st.columns(3)
    c1.metric("Ülke sayısı", len(df))
    c2.metric("Gösterge sayısı", len(secili_gostergeler))
    c3.metric("Anomali (aykırı ekonomi)", int((df["Anomali"] == -1).sum()))
    st.dataframe(
        df[["ulke"] + secili_gostergeler].rename(columns=ETIKET),
        use_container_width=True, height=380,
    )
    st.markdown("**Tanımlayıcı istatistikler**")
    st.dataframe(df[secili_gostergeler].rename(columns=ETIKET).describe().round(2), use_container_width=True)

    g = st.selectbox("Dağılımını incele", secili_gostergeler, format_func=lambda x: ETIKET[x])
    fig = px.histogram(df, x=g, nbins=25, labels={g: ETIKET[g]}, color_discrete_sequence=["#2563eb"])
    st.plotly_chart(fig, use_container_width=True)

# --- 2) Korelasyon -----------------------------------------------------------
with sekme2:
    st.subheader("Göstergeler Arası Korelasyon Analizi")
    kor = df[secili_gostergeler].corr().rename(index=ETIKET, columns=ETIKET)
    fig = px.imshow(kor, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(height=550)
    st.plotly_chart(fig, use_container_width=True)
    en_guclu = kor.where(~np.eye(len(kor), dtype=bool)).abs().stack().idxmax()
    st.success(f"En güçlü ilişki: **{en_guclu[0]} ↔ {en_guclu[1]}**")

# --- 3) Kümeleme -------------------------------------------------------------
with sekme3:
    st.subheader(f"K-Means Kümeleme (K={k}) — PCA ile 2 Boyutlu Görselleştirme")
    df["Küme Adı"] = "Küme " + df["Küme"].astype(str)
    fig = px.scatter(
        df, x="PC1", y="PC2", color="Küme Adı", text="ulke",
        hover_data={g: True for g in secili_gostergeler},
        labels={"PC1": f"Bileşen 1 (%{pca.explained_variance_ratio_[0]*100:.0f} varyans)",
                "PC2": f"Bileşen 2 (%{pca.explained_variance_ratio_[1]*100:.0f} varyans)"},
    )
    fig.update_traces(textposition="top center", textfont_size=9)
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Küme profilleri (ortalama değerler)**")
    kume_ozet = df.groupby("Küme")[secili_gostergeler].mean().round(1)
    st.dataframe(kume_ozet.rename(columns=ETIKET), use_container_width=True)

    aykirilar = df[df["Anomali"] == -1]["ulke"].tolist()
    if aykirilar:
        st.warning("⚠️ **Aykırı (anomali) ekonomiler:** " + ", ".join(aykirilar)
                   + " — Bu ülkeler genel örüntüden belirgin şekilde sapıyor (Isolation Forest).")

# --- 4) Ülke karşılaştırma ---------------------------------------------------
with sekme4:
    st.subheader("Ülke Karşılaştırma (Radar Grafiği)")
    secilen_ulkeler = st.multiselect(
        "Karşılaştırılacak ülkeler", df["ulke"].tolist(),
        default=[u for u in ["Türkiye", "Almanya", "Çin"] if u in df["ulke"].values],
    )
    if len(secilen_ulkeler) >= 2:
        # 0-100 normalize ederek radar çiz
        norm = (X_ham - X_ham.min()) / (X_ham.max() - X_ham.min()) * 100
        norm["ulke"] = df["ulke"].values
        fig = go.Figure()
        for u in secilen_ulkeler:
            satir = norm[norm["ulke"] == u][secili_gostergeler].iloc[0]
            fig.add_trace(go.Scatterpolar(
                r=satir.tolist() + [satir.tolist()[0]],
                theta=[ETIKET[g] for g in secili_gostergeler] + [ETIKET[secili_gostergeler[0]]],
                fill="toself", name=u,
            ))
        fig.update_layout(height=550, polar=dict(radialaxis=dict(range=[0, 100])))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            df[df["ulke"].isin(secilen_ulkeler)][["ulke"] + secili_gostergeler].rename(columns=ETIKET),
            use_container_width=True,
        )
    else:
        st.info("Lütfen en az 2 ülke seçin.")

# --- 5) AI Analist -----------------------------------------------------------
with sekme5:
    st.subheader("🤖 Yapay Zekâ Ekonomi Analisti (Google Gemini)")
    st.markdown(
        "Yapay zekâ; kümeleme sonuçlarını, küme profillerini ve aykırı ekonomileri okuyarak "
        "bir **ekonomist gözüyle** yorum raporu üretir."
    )
    soru = st.text_input(
        "İsteğe bağlı: AI analiste özel bir soru sorun",
        placeholder="Örn: Türkiye hangi kümede ve bu ne anlama geliyor?",
    )
    if st.button("📝 AI Analiz Raporu Üret", type="primary"):
        kume_ozet = df.groupby("Küme")[secili_gostergeler].mean().round(1)
        uyelikler = df.groupby("Küme")["ulke"].apply(list).to_dict()
        istem = f"""Sen deneyimli bir ekonomistsin. Aşağıda {len(df)} ülkenin Dünya Bankası 2023
göstergeleriyle yapılmış K-Means kümeleme (K={k}) analizi var.

Küme ortalama profilleri:
{kume_ozet.rename(columns=ETIKET).to_string()}

Küme üyelikleri:
{json.dumps({f'Küme {a}': b for a, b in uyelikler.items()}, ensure_ascii=False)}

Aykırı (anomali) tespit edilen ülkeler: {df[df['Anomali']==-1]['ulke'].tolist()}

{f'''Görev: Kullanıcı sana şu soruyu sordu: "{soru}"
Önce BU SORUYA doğrudan, net ve spesifik bir cevap ver (sorunun öznesi neyse ona odaklan,
genel rapor yazma). Cevabını verilerle destekle. En sonda 2-3 cümlelik kısa bir genel
değerlendirme ekleyebilirsin. Türkçe yaz, ~250 kelime.''' if soru else f'''Görev: Türkçe, akademik ama anlaşılır bir dille:
1) Her kümeyi ekonomik olarak isimlendir ve karakterize et,
2) Türkiye'nin konumunu özel olarak değerlendir,
3) Aykırı ülkelerin neden saptığını açıkla,
4) Politika çıkarımlarıyla bitir. Madde işaretleri kullan, ~350 kelime.'''}"""
        with st.spinner("Gemini analiz ediyor..."):
            yanit, hata = gemini_sor(istem)
        if yanit:
            st.markdown(yanit)
            st.session_state["son_ai_rapor"] = yanit
        else:
            st.info("Gemini API'ye ulaşılamadı; yerleşik analiz motoru kullanıldı.")
            if hata:
                st.caption(f"Teknik detay: {hata}")
            st.markdown(yerlesik_yorum(df, kume_ozet))

# --- 6) Google Sheets --------------------------------------------------------
with sekme6:
    st.subheader("📤 Google Sheets Entegrasyonu")
    st.markdown(
        "Kümeleme sonuçları ve küme profilleri tek tıkla Google Sheets'e aktarılır; "
        "böylece analiz çıktıları bulutta saklanır ve paylaşılabilir."
    )
    sonuc_df = df[["ulke", "iso3"] + secili_gostergeler + ["Küme"]].copy()
    sonuc_df["Analiz Tarihi"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    if st.button("☁️ Sonuçları Google Sheets'e Kaydet"):
        ok, mesaj = sheets_kaydet(sonuc_df, "EkonoMiner_Sonuclar")
        (st.success if ok else st.warning)(mesaj)
        if not ok:
            st.caption("Servis hesabı yapılandırılmamışsa aşağıdan CSV indirebilirsiniz.")
    st.download_button(
        "⬇️ Sonuçları CSV olarak indir",
        sonuc_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="ekonominer_sonuclar.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "EkonoMiner • Veri: World Bank Open Data • AI: Google Gemini • "
    "İKT-442 Veri Madenciliği Final Projesi — Affan Aybars Damgacı (230502050175)"
)
