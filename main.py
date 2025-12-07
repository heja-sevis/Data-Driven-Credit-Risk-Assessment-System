import streamlit as st
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Sayfa Yapılandırması
st.set_page_config(
    page_title="💳 Kredi Onayı Modelleri Analiz Platformu", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------
# 1. VERİ YÜKLEME VE MODEL EĞİTİMİ (Arka Plan - YALNIZCA BİR KEZ ÇALIŞIR)
# ----------------------------------------------------------------------

@st.cache_resource(show_spinner="⏳ Veri yükleniyor ve tüm 6 model eğitiliyor...")
def load_data_and_train_models():
    """Tüm veriyi yükler, ön işler, eğitir ve sonuçları döndürür."""
    
    try:
        credit_approval = fetch_ucirepo(id=27)
        X = credit_approval.data.features
        y = credit_approval.data.targets
        
    except Exception as e:
        st.error(f"❌ Veri yüklenirken hata: {e}")
        return None, None, None
    
    # Ön İşleme (Label Encoding)
    X_processed = X.copy()
    categorical_columns = X_processed.select_dtypes(include=['object']).columns

    for col in categorical_columns:
        le = LabelEncoder()
        X_processed[col] = le.fit_transform(X_processed[col].astype(str))

    if isinstance(y, pd.DataFrame):
        y = y.squeeze()
    if y.dtype == 'object' or isinstance(y.iloc[0], str):
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)

    # Split, Scaling, Imputation
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    imputer = SimpleImputer(strategy='mean')
    X_train_final = imputer.fit_transform(X_train_scaled)
    X_test_final = imputer.transform(X_test_scaled)
    
    # Model Eğitimi
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Support Vector Machine (SVM)": SVC(random_state=42),
        "Gradient Boosting Machines (GBM)": GradientBoostingClassifier(random_state=42),
        "Neural Network (MLP)": MLPClassifier(random_state=42, max_iter=300)
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train_final, y_train)
        y_pred = model.predict(X_test_final)
        
        report = classification_report(y_test, y_pred, output_dict=True)
        
        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "report": report,
            "conf_matrix": confusion_matrix(y_test, y_pred),
            "precision": report['weighted avg']['precision'],
            "recall": report['weighted avg']['recall'],
        }
    
    X_df = pd.DataFrame(X_processed, columns=X.columns)
    return results, X_df, credit_approval.metadata


# ----------------------------------------------------------------------
# 2. SAYFA FONKSİYONLARI
# ----------------------------------------------------------------------

def show_data_prep_page(X_raw, metadata, results):
    """Veri Hazırlığı ve Giriş sayfasını gösterir ve tüm model skorlarını karşılaştırır."""
    
    st.title("📚 Veri Seti İncelemesi ve Ön İşleme Adımları")
    
    # --- Veri Seti Özeti ---
    st.header("1️⃣ Ön İşleme Yapılmış Veri Seti Ön İzlemesi")
    st.info(f"Toplam örnek sayısı: **{X_raw.shape[0]}**, Özellik sayısı: **{X_raw.shape[1]}**")
    
    st.dataframe(X_raw.head(10), use_container_width=True)

    # --- Ön İşleme Adımları ---
    st.header("2️⃣ Uygulanan Veri Hazırlık Süreci")
    col_prep, col_info = st.columns(2)
    
    with col_prep:
        st.markdown("""
        * **Veri Kaynağı:** UCI Machine Learning Repository (Credit Approval).
        * **Kategorik Dönüşüm:** **Label Encoding** uygulandı (**data Subset.ipynb**).
        * **Eksik Değerler:** **Ortalama (Mean) Imputation** ile dolduruldu (**data Imputation.ipynb**).
        * **Özellik Ölçekleme:** `StandardScaler` ile tüm değerler normalize edildi.
        * **Bölme:** Eğitim (%70) ve Test (%30) olarak ayrıldı.
        """)

    with col_info:
        st.subheader("Veri Seti Metadataları")
        # Metadata'dan önemli bilgileri çekip listeleyelim
        if metadata and 'num_instances' in metadata and 'num_features' in metadata:
            st.markdown(f"**Örnek Sayısı:** {metadata['num_instances']}")
            st.markdown(f"**Özellik Sayısı:** {metadata['num_features']}")
            st.markdown(f"**Alan:** {metadata['area']}")
            st.markdown(f"**Özet:** {metadata['abstract'][:150]}...")
    
    st.write("---")
    
    # --- Tüm Modellerin Karşılaştırması (Görsel) ---
    st.header("3️⃣ Modellerin Genel Doğruluk Karşılaştırması")
    
    all_accuracies = {name: res['accuracy'] for name, res in results.items()}
    accuracy_df = pd.DataFrame(all_accuracies.items(), columns=['Model', 'Doğruluk Skoru'])
    
    # Görselleştirme
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x='Model', y='Doğruluk Skoru', data=accuracy_df.sort_values(by='Doğruluk Skoru', ascending=False), palette='viridis', ax=ax)
    
    ax.set_title("Farklı Sınıflandırıcıların Doğruluk Skorları")
    ax.set_ylabel("Doğruluk (Accuracy)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)


def show_model_comparison_page(results):
    """Model Karşılaştırma ve Sonuçlar sayfasını gösterir."""
    st.title("📈 Model Performans Değerlendirmesi")
    st.markdown("Eğitilmiş modellerden birini seçerek detaylı metriklerini inceleyin.")
    
    # --- Sidebar Model Seçimi ---
    st.sidebar.header("🎯 Model Seçimi")
    model_name = st.sidebar.selectbox(
        "İncelenecek Modeli Seçin:",
        list(results.keys()),
        index=2 
    )

    selected_result = results[model_name]

    st.header(f"Seçilen Model: **{model_name}**")
    st.write("---")

    # --- 1. Temel Metrikler (Metrik Kartları) ---
    st.subheader("1. Temel Performans Metrikleri")
    
    col_acc, col_prec, col_rec = st.columns(3)
    
    # Doğruluk Kartı
    with col_acc:
        st.metric(label="✅ Doğruluk (Accuracy)", 
                  value=f"{selected_result['accuracy']:.4f}",
                  delta=None) # Delta değeri, önceki sayfadaki en iyi modelle karşılaştırma için kullanılabilir.
    
    # Kesinlik (Precision) Kartı
    with col_prec:
        st.metric(label="🔍 Ortalama Kesinlik (Precision)",
                  value=f"{selected_result['precision']:.4f}")
    
    # Geri Çağırma (Recall) Kartı
    with col_rec:
        st.metric(label="🔄 Ortalama Geri Çağırma (Recall)",
                  value=f"{selected_result['recall']:.4f}")

    st.write("---")

    # --- 2. Sınıflandırma Raporu ve Karmaşıklık Matrisi ---
    st.subheader("2. Detaylı Metrik Analizi")
    
    col_report, col_matrix = st.columns(2)
    
    # Sınıflandırma Raporu
    with col_report:
        st.markdown("##### 📄 Sınıflandırma Raporu")
        report_df = pd.DataFrame(selected_result['report']).transpose()
        # Sayısal formatı düzenleme
        for col in ['precision', 'recall', 'f1-score']:
            if col in report_df.columns:
                 report_df[col] = report_df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
                 
        st.dataframe(report_df.style.highlight_max(axis=0, color='lightgreen', subset=pd.IndexSlice[['0', '1'], ['precision', 'recall', 'f1-score']]), 
                     use_container_width=True)
        st.caption("Not: Rapor, ağırlıklı ortalama (weighted avg) değerleri içermektedir.")


    # Karmaşıklık Matrisi
    with col_matrix:
        st.markdown("##### 📉 Karmaşıklık Matrisi (Confusion Matrix)")
        
        fig, ax = plt.subplots(figsize=(6, 5))
        cm = selected_result['conf_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Reddedildi (0)', 'Onaylandı (1)'], 
                    yticklabels=['Reddedildi (0)', 'Onaylandı (1)'],
                    ax=ax)
        ax.set_title(f"{model_name} Matrisi")
        ax.set_xlabel("Tahmin Edilen")
        ax.set_ylabel("Gerçek")
        st.pyplot(fig)


# ----------------------------------------------------------------------
# 3. ANA UYGULAMA MANTIĞI
# ----------------------------------------------------------------------

def main():
    
    # 1. Veri Yükleme ve Modelleri Eğitme
    results, X_raw, metadata = load_data_and_train_models()
    
    if results is None:
        return

    # 2. Sayfa Seçimi (Sidebar)
    PAGES = {
        "📊 Veri Hazırlığı ve Genel Karşılaştırma": show_data_prep_page,
        "🏆 Model Detay ve Metrikler": show_model_comparison_page,
    }

    st.sidebar.title("Credit Approval Analizi")
    st.sidebar.markdown("---")
    
    selection = st.sidebar.radio("Sayfa Seçimi", list(PAGES.keys()))
    st.sidebar.markdown("---")
    st.sidebar.success("✅ Veri ve Modeller Hazır!")
    
    # 3. Seçilen Sayfayı Göster
    if selection == "📊 Veri Hazırlığı ve Genel Karşılaştırma":
        PAGES[selection](X_raw, metadata, results) # results'ı karşılaştırma için gönderdik
    elif selection == "🏆 Model Detay ve Metrikler":
        PAGES[selection](results)

if __name__ == "__main__":
    main()
