import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

st.title("📈 Trend Post Bashoratchi")
st.write("Likes, Shares, Comments, Views sonlariga asoslanib trend score va trend darajasini aniqlang.")

# Ma'lumotni o‘qish
data = pd.read_csv('Viral_Social_Media_Trends.csv')

# Trend Score ustunini yaratish
data['Trend_Score'] = (
    data['Likes'].fillna(0) * 0.4 +
    data['Shares'].fillna(0) * 0.3 +
    data['Comments'].fillna(0) * 0.2 +
    data['Views'].fillna(0) * 0.1
)

# Sidebar bo‘limlarini yaratish
tabs = st.sidebar.radio("Bo‘limlar", ('Trend Score Bashorati', 'Hashtaglar va Platformalar', 'KMeans Guruhlar', 'Mintaqa va Hashtag Ko‘rishlar','Dataset'))

# === Trend Score Bashorati ===
if tabs == 'Trend Score Bashorati':
    st.subheader("📊 Trend Score Bashorat qilish")

    # Model qurish
    X = data[['Likes', 'Shares', 'Comments', 'Views']].fillna(0)
    y = data['Trend_Score']
    model = LinearRegression()
    model.fit(X, y)

    # Trend darajasini aniqlash
    def classify_trend(score):
        if score >= 700:
            return "🔥 High"
        elif score >= 300:
            return "⚡ Medium"
        else:
            return "🌱 Low"

    # Foydalanuvchi inputlari
    likes = st.number_input("Likes soni", min_value=0)
    shares = st.number_input("Shares soni", min_value=0)
    comments = st.number_input("Comments soni", min_value=0)
    views = st.number_input("Views soni", min_value=0)

    if st.button("Trend Score ni Bashorat qilish"):
        input_data = pd.DataFrame([[likes, shares, comments, views]],
                                  columns=['Likes', 'Shares', 'Comments', 'Views'])
        predicted_score = model.predict(input_data)[0]
        level = classify_trend(predicted_score)

        st.success(f"🔢 Bashorat qilingan Trend Score: {round(predicted_score, 2)}")
        st.info(f"📊 Trend Darajasi: {level}")

# === Hashtaglar va Platformalar ===
elif tabs == 'Hashtaglar va Platformalar':
    st.subheader("📊 Hashtaglar va platformalar bo'ylab jalb qilish darajalari")

    # 'Platform', 'Hashtag', 'Engagement_Level' ustunlari bo‘yicha guruhlash va engagement sonini hisoblash
    engagement_counts = data.groupby(['Platform', 'Hashtag', 'Engagement_Level']).size().unstack().fillna(0)

    # Stacked bar chart chizish
    fig, ax = plt.subplots(figsize=(8.5, 4))
    engagement_counts.plot(kind='bar', stacked=True, ax=ax)

    # Yangi xususiyatlar qo‘shish
    plt.title('Hashtaglar va platformalar bo`ylab jalb qilish darajalari')
    plt.ylabel('Postlar soni')
    plt.xticks(rotation=90, ha='right')

    # Bar chartni ko‘rsatish
    st.pyplot(fig)

# === KMeans Guruhlar ===
# elif tabs == 'KMeans Guruhlar':
#     st.subheader("📊 KMeans Guruhlar Bo‘yicha O‘rtacha Atributlar")

#     # Features tanlash (Likes, Shares, Comments, Views)
#     features = data[['Likes', 'Shares', 'Comments', 'Views']].fillna(0)

#     # KMeans modelini yaratish (5 ta guruh)
#     kmeans = KMeans(n_clusters=5, random_state=42)
#     data['Cluster'] = kmeans.fit_predict(features)

#     # KMeans guruhlar bo‘yicha o‘rtacha qiymatlarni hisoblash
#     cluster_means = data.groupby('Cluster')[['Likes', 'Shares', 'Comments', 'Views']].mean()

#     # Bar chart chizish
#     fig, ax = plt.subplots(figsize=(8, 5))
#     cluster_means.plot(kind='bar', ax=ax, color=['#FE7331', '#344054', '#6C757D', '#D6D9E0'])

#     plt.title('KMeans Guruhlar Bo‘yicha O‘rtacha Atributlar')
#     plt.xlabel('Cluster')
#     plt.ylabel('O‘rtacha Qiymat')
#     plt.xticks(rotation=0)

#     st.pyplot(fig)

# === Mintaqa va Hashtag Ko‘rishlar ===
elif tabs == 'Mintaqa va Hashtag Ko‘rishlar':
    st.subheader("🌍 Issiqlik xaritasi: Mintaqa va Hashtag bo‘yicha o‘rtacha ko‘rishlar soni")

    # Pivot jadval tayyorlash
    region_engagement = data.pivot_table(
        index='Region',
        columns='Hashtag',
        values='Views',
        aggfunc='mean'
    ).fillna(0)

    # Heatmap chizish
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(region_engagement, cmap='YlGnBu', linewidths=0.5, ax=ax)

    plt.title('Heatmap: Mintaqa va Hashtag bo‘yicha Views')
    plt.xlabel('Hashtag')
    plt.ylabel('Mintaqa')
    st.pyplot(fig)


elif tabs == 'Dataset':
    st.subheader("📊 Dataset")
    st.write(data.head())
    

# === KMeans Guruhlar ===
elif tabs == 'KMeans Guruhlar':
    st.subheader("📊 KMeans Guruhlar Bo‘yicha O‘rtacha Atributlar")

    # Guruhlar sonini tanlash uchun slider
    n_clusters = st.slider("Guruhlar soni (K)", min_value=2, max_value=10, value=5, step=1)

    # Features tanlash (Likes, Shares, Comments, Views)
    features = data[['Likes', 'Shares', 'Comments', 'Views']].fillna(0)

    # KMeans modelini yaratish
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['Cluster'] = kmeans.fit_predict(features)

    # KMeans guruhlar bo‘yicha o‘rtacha qiymatlarni hisoblash
    cluster_means = data.groupby('Cluster')[['Likes', 'Shares', 'Comments', 'Views']].mean()

    # Bar chart chizish
    fig, ax = plt.subplots(figsize=(8, 5))
    cluster_means.plot(kind='bar', ax=ax, color=['#FE7331', '#344054', '#6C757D', '#D6D9E0'])

    plt.title(f'KMeans ({n_clusters} Guruh) Bo‘yicha O‘rtacha Atributlar')
    plt.xlabel('Cluster')
    plt.ylabel('O‘rtacha Qiymat')
    plt.xticks(rotation=0)

    # Vizualni ko‘rsatish
    st.pyplot(fig)

    # Guruhdagi postlar sonini ko‘rsatish
    st.subheader("📊 Guruhlar bo‘yicha postlar soni")
    cluster_counts = data['Cluster'].value_counts().sort_index()
    st.bar_chart(cluster_counts)
