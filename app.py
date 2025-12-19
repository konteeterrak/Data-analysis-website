import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
st.set_page_config(
    page_title="Interactive Streamlit Dashboard",
    layout="wide"
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
/* Background */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

/* Card style */
.card {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.card:hover {
    transform: translateY(-8px) scale(1.02);
    box-shadow: 0 12px 40px rgba(0,0,0,0.6);
}

/* Button animation */
.stButton>button {
    border-radius: 12px;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    transform: scale(1.05);
    background-color: #00c6ff;
}

/* Fade animation */
.fade-in {
    animation: fadeIn 1.2s ease-in;
}
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(20px);}
    to {opacity: 1; transform: translateY(0);}
}
</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown("""
<div class="fade-in">
    <h1 style="text-align:center;">Streamlit Interactive Dashboard</h1>
    <p style="text-align:center;">Beautiful • Animated • Interactive</p>
</div>
""", unsafe_allow_html=True)



# Button
st.button("Simple Button", type="primary")

# Checkbox
if st.checkbox('Show dataframe'):
    chart_data = pd.DataFrame(
       np.random.randn(20, 3),
       columns=['a', 'b', 'c'])

    chart_data

# Selectbox
option = st.selectbox('Which major do you like best?', ['CO', 'CI'])
st.write("You've selected: ", option)

# แสดงข้อมูลจาก dataframe
df = pd.read_csv('sustainable_waste_management_dataset_2024.csv')
st.dataframe(df) # st.write(df)
# st.table(df)

from numpy.random import default_rng as rng
# ตัวอย่างกราฟแท่ง จาก Docs
df = pd.DataFrame(rng(0).standard_normal((20, 3)), columns=["a", "b", "c"])
st.bar_chart(df)

# ตัวอย่างกราฟเส้น จาก Docs
df = pd.DataFrame(rng(0).standard_normal((20, 3)), columns=["a", "b", "c"])
st.line_chart(df)

# Column Layout
col1, col2, col3 = st.columns(3)

col1.header("BAR CHART")

col2.header("LINE CHART")

col3.header("AREA CHART")


# หรือใช้กับคำว่า "with" ได้
df_chart = pd.DataFrame(rng(0).standard_normal((20, 3)), columns=["a", "b", "c"])
with col1:
    st.bar_chart(df_chart)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.line_chart(df_chart)
    st.markdown('</div>', unsafe_allow_html=True) 

with col3:
    st.area_chart(df_chart)
    st.markdown('</div>', unsafe_allow_html=True) 

xd = st.text_input('plaese write your file name: ')
# Container Layout
container = st.container(border=True)
container.write(f"This is a file name : {xd}")

container.write(f"This is a information about flie name : {xd}")



# อัพโหลดไฟล์
file = st.file_uploader("Upload a CSV For Preview a file")

# สามารถเขียนโค้ดเพื่อ extract ข้อมูลออกมาได้
if file:
    df = pd.read_csv(file)
    st.dataframe(df, use_container_width=True)

    if set(['age','major','year']).issubset(df.columns):
        st.bar_chart(df, x='year', y='age', color='major', stack=False)

st.header("🤖 Train Linear Regression from Uploaded CSV")

# ------------------ Upload CSV ------------------




# ------------------ Helper ------------------
def remove_outliers_iqr(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    return df[~((df < (Q1 - 1.5 * IQR)) | 
                (df > (Q3 + 1.5 * IQR))).any(axis=1)]

# ------------------ Upload ------------------
file = st.file_uploader("📂 Upload ANY CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.subheader("📄 Raw Data")
    st.dataframe(df, use_container_width=True)

    # ------------------ Numeric only ------------------
    df = df.select_dtypes(include=np.number)

    if df.shape[1] < 2:
        st.error("❌ Need at least 2 numeric columns")
        st.stop()

    # ------------------ Cleaning options ------------------
    st.subheader("🧹 Data Cleaning Options")
    remove_outlier = st.checkbox("Remove outliers (IQR)", value=True)
    use_log = st.checkbox("Log transform target")

    df = df.dropna()
    if remove_outlier:
        df = remove_outliers_iqr(df)

    st.success(f"✅ Cleaned samples: {len(df)}")

    # ------------------ Target ------------------
    target = st.selectbox("🎯 Select Target (Y)", df.columns)
    features = [c for c in df.columns if c != target]

    X = df[features]
    y = df[target]

    if use_log:
        y = np.log1p(y)

    # ------------------ Train ------------------
    if st.button("🚀 Train Model"):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # ------------------ Metrics ------------------
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        col1, col2 = st.columns(2)
        col1.metric("📉 MSE", f"{mse:.4f}")
        col2.metric("📈 R²", f"{r2:.4f}")

        # ------------------ Plot ------------------
        st.subheader("📊 Predicted vs Actual")

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_test, y_pred, alpha=0.7)
        ax.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            '--'
        )
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.grid(True)
        st.pyplot(fig)

        # ------------------ Coef ------------------
        st.subheader("🧮 Model Coefficients")
        coef_df = pd.DataFrame({
            "Feature": features,
            "Coefficient": model.coef_
        }).sort_values("Coefficient", key=abs, ascending=False)

        st.dataframe(coef_df, use_container_width=True)

st.markdown("""
<hr>
<p style="text-align:center; opacity:0.7;">
Made with getnoz using Streamlit
</p>
""", unsafe_allow_html=True)