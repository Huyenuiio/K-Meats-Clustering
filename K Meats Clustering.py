import os  # Thư viện os để thực hiện các lệnh hệ thống
import pandas as pd  # Thư viện pandas để xử lý dữ liệu
from sklearn.cluster import KMeans  # Thư viện sklearn để thực hiện KMeans clustering
from sklearn.preprocessing import StandardScaler  # Thư viện sklearn để chuẩn hóa dữ liệu
from sklearn.impute import KNNImputer  # Thư viện sklearn để xử lý giá trị khuyết bằng KNN Imputer
from sklearn.decomposition import PCA  # Thư viện sklearn để giảm chiều dữ liệu bằng PCA
import matplotlib.pyplot as plt  # Thư viện matplotlib để vẽ biểu đồ
import seaborn as sns  # Thư viện seaborn để vẽ biểu đồ
import pywaffle as Waffle  # Thư viện pywaffle để vẽ biểu đồ Waffle
import numpy as np  # Thư viện numpy để xử lý mảng
from sklearn.metrics import silhouette_samples, silhouette_score  # Thư viện sklearn để tính toán silhouette score

try:
    from pywaffle import Waffle  # Thử import thư viện pywaffle
    HAS_PYWAFFLE = True  # Nếu thành công, đặt biến HAS_PYWAFFLE là True
except ModuleNotFoundError:
    HAS_PYWAFFLE = False  # Nếu không thành công, đặt biến HAS_PYWAFFLE là False
    print("Module 'pywaffle' is not installed. Waffle chart will not be displayed.")  # Thông báo lỗi nếu không cài đặt được pywaffle

data = 'CC GENERAL.csv'  # Đường dẫn tới file dữ liệu
df = pd.read_csv(data)  # Đọc dữ liệu từ file CSV

def intro():     
    os.system("clear")  # Clear the console screen
    print("""\033[1;36m
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║\033[1;34m ██╗  ██╗      ███╗   ███╗███████╗ █████╗ ███╗   ██╗███████╗     ██████╗██╗     ██╗   ██╗███████╗████████╗███████╗██████╗ ██╗███╗   ██╗ ██████╗   \033[1;36m║
║\033[1;35m ██║ ██╔╝      ████╗ ████║██╔════╝██╔══██╗████╗  ██║██╔════╝    ██╔════╝██║     ██║   ██║██╔════╝╚══██╔══╝██╔════╝██╔══██╗██║████╗  ██║██╔════╝   \033[1;36m║
║\033[1;36m █████╔╝       ██╔████╔██║█████╗  ███████║██╔██╗ ██║███████╗    ██║     ██║     ██║   ██║███████╗   ██║   █████╗  ██████╔╝██║██╔██╗ ██║██║  ███╗  \033[1;36m║
║\033[1;34m ██╔═██╗       ██║╚██╔╝██║██╔══╝  ██╔══██║██║╚██╗██║╚════██║    ██║     ██║     ██║   ██║╚════██║   ██║   ██╔══╝  ██╔══██╗██║██║╚██╗██║██║   ██║  \033[1;36m║
║\033[1;35m ██║  ██╗      ██║ ╚═╝ ██║███████╗██║  ██║██║ ╚████║███████║    ╚██████╗███████╗╚██████╔╝███████║   ██║   ███████╗██║  ██║██║██║ ╚████║╚██████╔╝  \033[1;36m║
║\033[1;36m ╚═╝  ╚═╝      ╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝     ╚═════╝╚══════╝ ╚═════╝ ╚══════╝   ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝ ╚═════╝   \033[1;36m║
║                                                                                                                                     Dev by Nhóm 1║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

\033[1;36m
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║\033[1;33m (1) Tiền xử lý dữ liệu                                                                                                                           \033[1;36m║
║\033[1;33m (2) Phân tích thăm dò (EDA)                                                                                                                      \033[1;36m║
║\033[1;33m (3) Data Normalization                                                                                                                           \033[1;36m║
║\033[1;33m (4) K-Means Clustering Analysis                                                                                                                  \033[1;36m║
║\033[1;32m                                                                                                                                                  \033[1;36m║
║\033[1;33m (0) About                                                                                                                                        \033[1;36m║
║\033[1;33m (00) Exit                                                                                                                                        \033[1;36m║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
\033[0m""")
    print("\n\033[1;37mPlease enter your choice: \033[0m")  # Display menu and prompt user for input
    try:
        choice = int(input(""))  # Read user input
        if choice == 4:
            main()  # Call main function if choice is 4
        elif choice == 0:
            print("\033[1;36m")
            print("""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                                                                                  ║
║\033[1;34m                                                                                                                                                  \033[1;36m║
║\033[1;35m                                                                                                                                                  \033[1;36m║
║                                                                                                                                                  ║
║\033[1;33m                                             *---Dev by To Van Huyen 226900---*                                                                   \033[1;36m║
║\033[1;33m                                                                                                                                                  \033[1;36m║
║                                                                                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
""")  # Display about information if choice is 0
            print("\033[0m")
            print("\033[0m")
        elif choice == 00:
            exit()  # Exit program if choice is 00
        elif choice == 1:
            data_description()  # Call data_description function if choice is 1
        elif choice == 2:
            exploratory_data_analysis()  # Call exploratory_data_analysis function if choice is 2
        elif choice == 3:
            data_normalization()  # Call data_normalization function if choice is 3
        else:
            print("\033[1;31mInvalid choice. Returning to main menu.\033[0m")  # Display error message for invalid choice
            intro()  # Call intro function to display menu again
    except ValueError:
        print("\033[1;31mPlease enter a valid number.\033[0m")  # Display error message for non-integer input
        intro()  # Call intro function to display menu again

def data_description():
    print(df.info())  # Hiển thị thông tin về DataFrame
    print(df.isnull().sum())  # Hiển thị số lượng giá trị khuyết trong mỗi cột
    while True:
        check_opjeck = input("\n Nhập giá trị mà bạn muốn kiểm tra (No): ")  # Yêu cầu người dùng nhập tên cột để kiểm tra
        if check_opjeck.lower() == 'no':
            break  # Nếu người dùng nhập 'no', thoát vòng lặp
        unique_count = len(df[check_opjeck].unique())  # Đếm số lượng giá trị duy nhất trong cột
        print(f"Column '{check_opjeck}' has {unique_count} unique values.")  # Hiển thị số lượng giá trị duy nhất
        print(df.describe())  # Hiển thị thống kê mô tả của DataFrame
        confirm = input(f"Xác nhận xóa thuộc tính '{check_opjeck}' không? (yes/no): ")  # Yêu cầu người dùng xác nhận xóa cột
        if confirm.lower() == 'yes':
            df.drop(columns=[check_opjeck], inplace=True)  # Xóa cột khỏi DataFrame
            df.to_csv('CC GENERAL.csv', index=False)  # Lưu DataFrame vào file CSV
            print(f"Column '{check_opjeck}' has been removed.")  # Hiển thị thông báo đã xóa cột
        break
    while True:
        choice = input("\nXử lý giá trị khuyết bằng KNN Imputer? (yes/no): ")  # Yêu cầu người dùng xác nhận xử lý giá trị khuyết
        if choice.lower() == "yes":
            imputer = KNNImputer(n_neighbors=5)  # Tạo đối tượng KNN Imputer
            df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)  # Xử lý giá trị khuyết và chuyển đổi lại thành DataFrame
            df_imputed.to_csv('CC GENERAL.csv', index=False)  # Lưu DataFrame đã xử lý vào file CSV
            print("Missing values have been imputed using KNN Imputer.")  # Hiển thị thông báo đã xử lý giá trị khuyết
            break
        elif choice.lower() == "no":
            break
        else:
            print("Vui lòng nhập 'yes' hoặc 'no'.")  # Nếu người dùng nhập không hợp lệ, hiển thị thông báo lỗi
    input("\nPress Enter to return to the main menu...")  # Yêu cầu người dùng nhấn Enter để quay lại menu chính
    intro()  # Gọi lại hàm intro để hiển thị menu

def exploratory_data_analysis():
    data = pd.read_csv('CC GENERAL.csv')  # Đọc dữ liệu từ file CSV
    plt.figure(figsize=(10, 6))  # Tạo biểu đồ với kích thước 10x6
    plt.scatter(data['TENURE'], data['BALANCE'], alpha=0.5, label='Balance')  # Vẽ biểu đồ scatter của TENURE và BALANCE
    plt.scatter(data['TENURE'], data['CREDIT_LIMIT'], alpha=0.5, label='Credit Limit')  # Vẽ biểu đồ scatter của TENURE và CREDIT_LIMIT
    plt.xlabel('Tenure (months)')  # Đặt nhãn trục x
    plt.ylabel('Amount ($)')  # Đặt nhãn trục y
    plt.title('Scatter Plot of Balance and Credit Limit by Tenure')  # Đặt tiêu đề biểu đồ
    plt.legend()  # Hiển thị chú thích
    plt.show()  # Hiển thị biểu đồ
    while True:
        choice = input("\nXác nhận Thực hiện phân tích số lượng mua so với tổng giao dịch? (yes/no): ")  # Yêu cầu người dùng xác nhận phân tích
        if choice.lower() == "yes":
            analyze_purchases_vs_transactions(data)  # Gọi hàm analyze_purchases_vs_transactions để phân tích
            break
        elif choice.lower() == "no":
            break
        else:
            print("Vui lòng nhập 'yes' hoặc 'no'.")  # Nếu người dùng nhập không hợp lệ, hiển thị thông báo lỗi
    input("\nPress Enter to return to the main menu...")  # Yêu cầu người dùng nhấn Enter để quay lại menu chính
    intro()  # Gọi lại hàm intro để hiển thị menu

def analyze_purchases_vs_transactions(data):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))  # Tạo biểu đồ với 2 subplot, kích thước 18x6
    sns.scatterplot(ax=axes[0], x='PURCHASES', y='TENURE', data=data, label='Purchases', color='blue')  # Vẽ biểu đồ scatter của PURCHASES và TENURE
    axes[0].set_xlabel('Amount ($)')  # Đặt nhãn trục x cho subplot đầu tiên
    axes[0].set_ylabel('Tenure (months)')  # Đặt nhãn trục y cho subplot đầu tiên
    axes[0].set_title('Purchases by Tenure')  # Đặt tiêu đề cho subplot đầu tiên
    axes[0].legend()  # Hiển thị chú thích cho subplot đầu tiên
    sns.scatterplot(ax=axes[1], x='PURCHASES_TRX', y='TENURE', data=data, label='Total Transactions', color='red')  # Vẽ biểu đồ scatter của PURCHASES_TRX và TENURE
    axes[1].set_xlabel('Amount ($)')  # Đặt nhãn trục x cho subplot thứ hai
    axes[1].set_ylabel('Tenure (months)')  # Đặt nhãn trục y cho subplot thứ hai
    axes[1].set_title('Total Transactions by Tenure')  # Đặt tiêu đề cho subplot thứ hai
    axes[1].legend()  # Hiển thị chú thích cho subplot thứ hai
    plt.show()  # Hiển thị biểu đồ
    while True:
        choice = input("\nTiến hành phân tích sự tương quan của các thuộc tính? (yes/no): ")  # Yêu cầu người dùng xác nhận phân tích tương quan
        if choice.lower() == "yes":
            analyze_correlation(data)  # Gọi hàm analyze_correlation để phân tích tương quan
            break
        elif choice.lower() == "no":
            break
        else:
            print("Vui lòng nhập 'yes' hoặc 'no'.")  # Nếu người dùng nhập không hợp lệ, hiển thị thông báo lỗi
    input("\nPress Enter to return to the main menu...")  # Yêu cầu người dùng nhấn Enter để quay lại menu chính
    intro()  # Gọi lại hàm intro để hiển thị menu

def analyze_correlation(data):
    correlation_matrix = data.corr()  # Tính toán ma trận tương quan
    plt.figure(figsize=(12, 8))  # Tạo biểu đồ với kích thước 12x8
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)  # Vẽ biểu đồ heatmap của ma trận tương quan
    plt.title('Correlation Matrix of Credit Card Data')  # Đặt tiêu đề biểu đồ
    plt.show()  # Hiển thị biểu đồ
    for col in correlation_matrix.columns:
        for row in correlation_matrix.index:
            if col != row:
                corr_value = correlation_matrix.loc[row, col]  # Lấy giá trị tương quan giữa hai cột
                if -0.3 < corr_value < 0.3:
                    print(f"Độ tương quan giữa {row} và {col} không tốt: {corr_value:.2f}")  # Hiển thị thông báo nếu độ tương quan không tốt
                elif -0.7 < corr_value < -0.3 or 0.7 > corr_value > 0.3:
                    print(f"Độ tương quan giữa {row} và {col} tương đối tốt: {corr_value:.2f}")  # Hiển thị thông báo nếu độ tương quan tương đối tốt
                elif -1 < corr_value < -0.7 or 1 > corr_value > 0.7:
                    print(f"Độ tương quan giữa {row} và {col} rất tốt: {corr_value:.2f}")  # Hiển thị thông báo nếu độ tương quan rất tốt

def data_normalization():
    data = pd.read_csv('CC GENERAL.csv')  # Đọc dữ liệu từ file CSV
    print("\nData before normalization:")  # Hiển thị thông báo trước khi chuẩn hóa dữ liệu
    print(data.head())  # Hiển thị 5 dòng đầu tiên của DataFrame
    features = data[['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE']]  # Chọn các thuộc tính liên quan đến hành vi sử dụng thẻ tín dụng
    features.dropna(inplace=True)  # Loại bỏ dữ liệu thiếu
    scaler = StandardScaler()  # Tạo đối tượng StandardScaler
    features_scaled = scaler.fit_transform(features)  # Chuẩn hóa dữ liệu
    features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)  # Chuyển đổi lại thành DataFrame
    features_scaled_df.to_csv('CC GENERAL Normalized.csv', index=False)  # Lưu DataFrame đã chuẩn hóa vào file CSV
    print("\nData after normalization:")  # Hiển thị thông báo sau khi chuẩn hóa dữ liệu
    print(features_scaled_df.head())  # Hiển thị 5 dòng đầu tiên của DataFrame đã chuẩn hóa
    while True:
        choice = input("\nXác nhận giảm chiều dữ liệu bằng PCA? (yes/no): ")  # Yêu cầu người dùng xác nhận giảm chiều dữ liệu
        if choice.lower() == "yes":
            pca = PCA(n_components=2)  # Tạo đối tượng PCA với 2 thành phần chính
            features_pca = pca.fit_transform(features_scaled)  # Giảm chiều dữ liệu
            features_pca_df = pd.DataFrame(features_pca, columns=['PC1', 'PC2'])  # Chuyển đổi lại thành DataFrame
            features_pca_df.to_csv('CC GENERAL PCA.csv', index=False)  # Lưu DataFrame đã giảm chiều vào file CSV
            print("\nData after PCA:")  # Hiển thị thông báo sau khi giảm chiều dữ liệu
            print(features_pca_df.head())  # Hiển thị 5 dòng đầu tiên của DataFrame đã giảm chiều
            break
        elif choice.lower() == "no":
            break
        else:
            print("Vui lòng nhập 'yes' hoặc 'no'.")  # Nếu người dùng nhập không hợp lệ, hiển thị thông báo lỗi
    input("\nPress Enter to return to the main menu...")  # Yêu cầu người dùng nhấn Enter để quay lại menu chính
    intro()  # Gọi lại hàm intro để hiển thị menu

def choose_optimal_clusters():
    data = pd.read_csv('CC GENERAL.csv')  # Đọc dữ liệu từ file CSV
    features = data[['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE']]  # Chọn các thuộc tính liên quan đến hành vi sử dụng thẻ tín dụng
    features.dropna(inplace=True)  # Loại bỏ dữ liệu thiếu
    scaler = StandardScaler()  # Tạo đối tượng StandardScaler
    features_scaled = scaler.fit_transform(features)  # Chuẩn hóa dữ liệu
    wss = []  # Tạo danh sách rỗng để lưu trữ Within-cluster Sum of Squares
    K = range(1, 11)  # Tạo danh sách các giá trị k từ 1 đến 10
    for k in K:
        kmeans = KMeans(n_clusters=k)  # Tạo đối tượng KMeans với số cụm là k
        kmeans.fit(features_scaled)  # Huấn luyện mô hình KMeans
        wss.append(kmeans.inertia_)  # Thêm giá trị Within-cluster Sum of Squares vào danh sách
    plt.figure(figsize=(10, 6))  # Tạo biểu đồ với kích thước 10x6
    plt.plot(K, wss, 'bx-')  # Vẽ biểu đồ Elbow
    plt.axvline(x=4, color='r', linestyle='--')  # Vẽ đường thẳng đứng tại k=4
    plt.text(4.2, wss[3], 'Elbow at k=4', color='red')  # Thêm chú thích tại k=4
    plt.xlabel('Number of clusters')  # Đặt nhãn trục x
    plt.ylabel('Within-cluster Sum of Squares')  # Đặt nhãn trục y
    plt.title('Elbow Method For Optimal number of clusters')  # Đặt tiêu đề biểu đồ
    plt.show()  # Hiển thị biểu đồ
    optimal_clusters = None  # Khởi tạo biến optimal_clusters
    for i in range(1, len(wss) - 1):
        if wss[i] - wss[i+1] < wss[i-1] - wss[i]:  # Kiểm tra điều kiện để tìm số cụm tối ưu
            optimal_clusters = i + 1  # Gán giá trị số cụm tối ưu
            break
    if optimal_clusters is None:
        optimal_clusters = len(K)  # Nếu không tìm thấy số cụm tối ưu, gán giá trị bằng độ dài của K
    print(f"\nSố cụm tối ưu (Elbow method): {optimal_clusters}")  # Hiển thị số cụm tối ưu
    input("\nPress Enter to return to the main menu...")  # Yêu cầu người dùng nhấn Enter để quay lại menu chính
    intro()  # Gọi lại hàm intro để hiển thị menu

def main():
    print("K-Means Clustering")  # Hiển thị thông báo
    print("*---------AI-----------*")  # Hiển thị thông báo
    data = pd.read_csv('CC GENERAL PCA.csv')  # Đọc dữ liệu từ file CSV
    x = data  # Gán dữ liệu vào biến x
    wss = []  # Tạo danh sách rỗng để lưu trữ Within-cluster Sum of Squares
    K = range(1, 11)  # Tạo danh sách các giá trị k từ 1 đến 10
    for k in K:
        kmeans = KMeans(n_clusters=k)  # Tạo đối tượng KMeans với số cụm là k
        kmeans.fit(x)  # Huấn luyện mô hình KMeans
        wss.append(kmeans.inertia_)  # Thêm giá trị Within-cluster Sum of Squares vào danh sách
    plt.figure(figsize=(10, 6))  # Tạo biểu đồ với kích thước 10x6
    plt.plot(K, wss, 'bx-')  # Vẽ biểu đồ Elbow
    plt.axvline(x=4, color='r', linestyle='--')  # Vẽ đường thẳng đứng tại k=4
    plt.text(4.2, wss[3], 'Elbow at k=4', color='red')  # Thêm chú thích tại k=4
    plt.xlabel('Number of clusters')  # Đặt nhãn trục x
    plt.ylabel('Within-cluster Sum of Squares')  # Đặt nhãn trục y
    plt.title('Elbow Method For Optimal number of clusters')  # Đặt tiêu đề biểu đồ
    plt.show()  # Hiển thị biểu đồ
    while True:
        try:
            num_clusters = int(input("Nhập số cụm (0 để sử dụng số cụm tối ưu từ Elbow method): "))  # Yêu cầu người dùng nhập số cụm
            if num_clusters < 0:
                print("Số cụm phải là số nguyên không âm.")  # Hiển thị thông báo lỗi nếu số cụm không hợp lệ
            elif num_clusters == 0:
                num_clusters = 4  # Giả sử số cụm tối ưu là 4 (từ Elbow method)
                break
            else:
                break
        except ValueError:
            print("Số cụm phải là số nguyên.")  # Hiển thị thông báo lỗi nếu người dùng nhập không phải số nguyên
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)  # Tạo đối tượng KMeans với số cụm đã chọn
    kmeans.fit(x)  # Huấn luyện mô hình KMeans
    x['Cluster'] = kmeans.labels_  # Thêm cột kết quả phân cụm vào DataFrame
    print("\nKết quả phân cụm:")  # Hiển thị thông báo kết quả phân cụm
    print(x.head())  # Hiển thị 5 dòng đầu tiên của DataFrame đã phân cụm
    while True:
        pca_file_path = input("\nNhập đường dẫn tới file PCA (ví dụ: 'CC GENERAL PCA.csv'): ")  # Yêu cầu người dùng nhập đường dẫn tới file PCA
        try:
            x = pd.read_csv(pca_file_path)  # Đọc dữ liệu từ file PCA
            print("\nDữ liệu PCA:")  # Hiển thị thông báo dữ liệu PCA
            print(x.head())  # Hiển thị 5 dòng đầu tiên của DataFrame PCA
            break
        except FileNotFoundError:
            print(f"File '{pca_file_path}' không tồn tại. Vui lòng nhập lại đường dẫn chính xác.")  # Hiển thị thông báo lỗi nếu file không tồn tại
    while True:
        choice = input("\nXác nhận vẽ biểu đồ phân cụm? (yes/no): ")  # Yêu cầu người dùng xác nhận vẽ biểu đồ phân cụm
        if choice.lower() == "yes":
            kmeans_pca = KMeans(n_clusters=num_clusters, random_state=42)  # Tạo đối tượng KMeans với số cụm đã chọn
            kmeans_pca.fit(x)  # Huấn luyện mô hình KMeans
            x['Cluster'] = kmeans_pca.labels_  # Thêm cột kết quả phân cụm vào DataFrame
            plt.figure(figsize=(10, 6))  # Tạo biểu đồ với kích thước 10x6
            scatter = plt.scatter(x.iloc[:, 0], x.iloc[:, 1], c=x['Cluster'], cmap='viridis', alpha=0.6)  # Vẽ biểu đồ scatter của PCA
            centroids = kmeans_pca.cluster_centers_  # Lấy tọa độ các centroid
            plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='*', s=200, label='Centroids')  # Vẽ các centroid
            for i in range(num_clusters):
                plt.text(centroids[i, 0], centroids[i, 1], f'Cluster {i}', fontsize=12, ha='center', va='center', color='black')  # Chú thích các cụm
            plt.xlabel('PCA Component 1')  # Đặt nhãn trục x
            plt.ylabel('PCA Component 2')  # Đặt nhãn trục y
            plt.title(f'K-Means Clustering on PCA Data (k={num_clusters})')  # Đặt tiêu đề biểu đồ
            plt.colorbar(scatter, label='Cluster')  # Hiển thị thanh màu
            plt.legend()  # Hiển thị chú thích
            plt.show()  # Hiển thị biểu đồ
            silhouette_avg = silhouette_score(x, kmeans_pca.labels_)  # Tính toán silhouette score trung bình
            sample_silhouette_values = silhouette_samples(x, kmeans_pca.labels_)  # Tính toán silhouette score cho từng mẫu
            plt.figure(figsize=(10, 6))  # Tạo biểu đồ với kích thước 10x6
            y_lower = 10  # Khởi tạo biến y_lower
            for i in range(num_clusters):
                ith_cluster_silhouette_values = sample_silhouette_values[kmeans_pca.labels_ == i]  # Lấy các giá trị silhouette của cụm thứ i
                ith_cluster_silhouette_values.sort()  # Sắp xếp các giá trị silhouette
                size_cluster_i = ith_cluster_silhouette_values.shape[0]  # Lấy kích thước của cụm thứ i
                y_upper = y_lower + size_cluster_i  # Tính toán y_upper
                plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, alpha=0.7)  # Vẽ biểu đồ silhouette
                plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))  # Thêm chú thích cho cụm
                y_lower = y_upper + 10  # Cập nhật y_lower
            plt.title("Silhouette plot for the various clusters")  # Đặt tiêu đề biểu đồ
            plt.xlabel("Silhouette coefficient values")  # Đặt nhãn trục x
            plt.ylabel("Cluster label")  # Đặt nhãn trục y
            plt.axvline(x=silhouette_avg, color="red", linestyle="--")  # Vẽ đường thẳng đứng tại silhouette score trung bình
            plt.show()  # Hiển thị biểu đồ
            try:
                cluster_counts = x['Cluster'].value_counts(normalize=True).sort_index() * 100  # Tính toán tỷ lệ phần trăm của từng cụm
                plt.figure(FigureClass=Waffle, rows=5, values=cluster_counts, title={'label': 'Waffle Chart of Cluster Distribution (Percentage)', 'loc': 'center'}, labels=[f"Cluster {i} ({count:.1f}%)" for i, count in enumerate(cluster_counts)], legend={'loc': 'upper left', 'bbox_to_anchor': (1, 1)})  # Vẽ biểu đồ Waffle
                plt.show()  # Hiển thị biểu đồ
            except ModuleNotFoundError:
                print("Module 'pywaffle' is not installed. Please install it to view the Waffle chart.")  # Hiển thị thông báo lỗi nếu không cài đặt được pywaffle
            break
        elif choice.lower() == "no":
            break
        else:
            print("Vui lòng nhập 'yes' hoặc 'no'.")  # Nếu người dùng nhập không hợp lệ, hiển thị thông báo lỗi
    intro()  # Gọi lại hàm intro để hiển thị menu

if __name__ == "__main__":
    intro()  # Gọi hàm intro khi chạy chương trình