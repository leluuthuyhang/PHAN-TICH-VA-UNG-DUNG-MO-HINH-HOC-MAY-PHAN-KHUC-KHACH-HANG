import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Tiêu đề và giới thiệu
st.title("ỨNG DỤNG PHÂN KHÚC KHÁCH HÀNG")
st.write("""
    Tải lên tệp CSV hoặc Excel chứa dữ liệu khách hàng. Ứng dụng này sử dụng thuật toán K-Means để phân tích dữ liệu hành vi khách hàng.
""")

# Tải lên dữ liệu
uploaded_file = st.file_uploader("Chọn dữ liệu định dạng CSV hoặc Excel", type=['csv', 'xlsx'])

if uploaded_file is not None:
    # Lấy dữ liệu
    if uploaded_file.name.endswith('csv'):
        data = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('xlsx'):
        data = pd.read_excel(uploaded_file, engine='openpyxl')

    # Xử lý giá trị missing
    st.subheader('XỬ LÝ GIÁ TRỊ MISSING')
    st.write("Hình dạng dữ liệu thô:", data.shape)
    st.write("Số lượng giá trị bị thiếu trước khi xử lý:", data.isnull().sum().sum())
    
    # Xử lý giá trị thiếu (NaN)
    data.dropna(inplace=True)

    # Xác minh tính toàn vẹn của dữ liệu sau khi xử lý giá trị thiếu
    st.write("Số lượng giá trị bị thiếu sau khi xử lý:", data.isnull().sum().sum())

    # Hiển thị dữ liệu
    if st.checkbox('Hiển thị dữ liệu thô'):
        st.subheader('Dữ liệu thô')
        st.write(data)

    # Tiền xử lý dữ liệu
    st.subheader('TIỀN XỬ LÝ DỮ LIỆU')
    columns = data.columns.tolist()

    # Chuyển đổi giá trị chuỗi thành số nguyên
    for col in columns:
        if data[col].dtype == 'object':  # Kiểm tra thuộc tính các cột dữ liệu có phải là object/ chuỗi (string) không
            encoder = LabelEncoder()
            data[col] = encoder.fit_transform(data[col]) # Chuyển đổi dữ liệu thành dạng số

    selected_columns = st.multiselect('Lựa chọn đặc trưng (cột) để phân cụm', columns)

    if selected_columns:
        st.write(f"Các cột được lựa chọn để phân cụm: {selected_columns}")
        if len(selected_columns) >= 2:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data[selected_columns]) # Chuẩn hóa dữ liệu trong khoảng từ -1 đến 1

            # Phân cụm
            st.subheader('PHÂN CỤM')
            num_clusters = st.slider('Chọn số lượng cụm - K', 2, 10, 3)
            kmeans = KMeans(n_clusters=num_clusters)
            data['Cluster'] = kmeans.fit_predict(scaled_data)

            # Trực quan hóa các cụm
            st.subheader('TRỰC QUAN HÓA PHÂN CỤM')
            if len(selected_columns) >= 2:
                fig, ax = plt.subplots()
                sns.scatterplot(x=data[selected_columns[0]], y=data[selected_columns[1]], hue=data['Cluster'], palette='viridis', ax=ax)
                st.pyplot(fig)
            else:
                st.write("Vui lòng chọn ít nhất hai cột để trực quan hóa cụm.")
        else:
            st.write("Vui lòng chọn ít nhất hai cột để trực quan hóa cụm.")
    else:
        st.write("Vui lòng chọn các cột để phân cụm.")

    # Footer
    st.markdown("---")
    st.write("© 2025 [llthang7211]. All rights reserved.")