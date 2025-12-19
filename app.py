import streamlit as st
import pandas as pd
import numpy as np


st.set_page_config(
    page_title="Interactive Streamlit Dashboard",
    page_icon="🚀",
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
    <h1 style="text-align:center;">🚀 Streamlit Interactive Dashboard</h1>
    <p style="text-align:center;">Beautiful • Animated • Interactive</p>
</div>
""", unsafe_allow_html=True)


name = st.text_input("👤 What is your name?")
if name:
    st.success(f"Hello {name} 👋")
st.markdown('</div>', unsafe_allow_html=True)

# Button
st.button("Simple Button", type="primary")

if st.button("Click me!"):
    st.write("You clicked me!")

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

col1.header("Debian")

col2.header("Fedora")

col3.header("Kali")


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
st.write("This is outside the container")

container.write(f"This is a information about flie name : {xd}")

# อัพโหลดไฟล์
file = st.file_uploader("Upload a CSV")

# สามารถเขียนโค้ดเพื่อ extract ข้อมูลออกมาได้
if (file):
    df = pd.read_csv(file)
    st.write(df)
    df = df[['age', 'major', 'year']]
    st.bar_chart(df, x='year', y='age', color='major', stack=False)

st.markdown("""
<hr>
<p style="text-align:center; opacity:0.7;">
Made with getnoz using Streamlit
</p>
""", unsafe_allow_html=True)