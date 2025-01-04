import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pywaffle as Waffle
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score

try:
    from pywaffle import Waffle
    HAS_PYWAFFLE = True
except ModuleNotFoundError:
    HAS_PYWAFFLE = False
    print("Module 'pywaffle' is not installed. Waffle chart will not be displayed.")




data = 'CC GENERAL.csv'
df = pd.read_csv(data)



def intro():
    os.system("clear")  # Thay thế cmd bằng os.system 
    print("""\033[1;32m
----------------------------------------------------------------------------------------------------------------------------------------
 ██╗  ██╗      ███╗   ███╗███████╗ █████╗ ███╗   ██╗███████╗     ██████╗██╗     ██╗   ██╗███████╗████████╗███████╗██████╗ ██╗███╗   ██╗ ██████╗ 
 ██║ ██╔╝      ████╗ ████║██╔════╝██╔══██╗████╗  ██║██╔════╝    ██╔════╝██║     ██║   ██║██╔════╝╚══██╔══╝██╔════╝██╔══██╗██║████╗  ██║██╔════╝ 
 █████╔╝       ██╔████╔██║█████╗  ███████║██╔██╗ ██║███████╗    ██║     ██║     ██║   ██║███████╗   ██║   █████╗  ██████╔╝██║██╔██╗ ██║██║  ███╗
 ██╔═██╗       ██║╚██╔╝██║██╔══╝  ██╔══██║██║╚██╗██║╚════██║    ██║     ██║     ██║   ██║╚════██║   ██║   ██╔══╝  ██╔══██╗██║██║╚██╗██║██║   ██║
 ██║  ██╗      ██║ ╚═╝ ██║███████╗██║  ██║██║ ╚████║███████║    ╚██████╗███████╗╚██████╔╝███████║   ██║   ███████╗██║  ██║██║██║ ╚████║╚██████╔╝
 ╚═╝  ╚═╝      ╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝     ╚═════╝╚══════╝ ╚═════╝ ╚══════╝   ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝ ╚═════╝ 
                                                                                                               Clustering Analysis Tool dev by nhóm 1
---------------------------------------------------------------------------------------------------------------------------------------                                                         

(1) Tiền xử lý dữ liệu
(2) Phân tích thăm dò (EDA)
(3) Data Normalization
(4) K-Means Clustering Analysis 

(0) About
(00) Exit
-----------------------------------------------------------------------
""")
    print("\nEnter your choice here: ")
    try:
        choice = int(input(""))
        if choice == 4:
            main()
        elif choice == 0:
            print("""
This tool is developed for customer segmentation using K-Means Clustering.
Created by AI Assistant, aimed at providing insights from customer data.
                  
                  đây là đồ án của nhóm 1 
""")
        elif choice == 00:
            exit()
        elif choice == 1:
            data_description()
        elif choice == 2:
            exploratory_data_analysis()
        elif choice == 3:
            data_normalization()
       
        else:
            print("Invalid choice. Returning to main menu.")
            intro()
    except ValueError:
        print("Please enter a valid number.")
        intro()


def data_description():
    # print(df.head())
    print(df.info())
    print(df.isnull().sum())
    # print("\nData Description:")
    # print(data.describe(include='all'))
    # print("\nData Types:")
    # print(data.dtypes)
    while True:
        check_opjeck = input("\n Nhập Object mà bạn muốn kiểm tra (no): ")
        if check_opjeck.lower() == 'no':
            break
        unique_count = len(df[check_opjeck].unique())
        print(f"Column '{check_opjeck}' has {unique_count} unique values.")
        print(df.describe())  # Display the descriptive statistics of the DataFrame
        confirm = input(f"Bạn có muốn xóa thuộc tính '{check_opjeck}' không? (yes/no): ")
        if confirm.lower() == 'yes':
            df.drop(columns=[check_opjeck], inplace=True)
            df.to_csv('CC GENERAL.csv', index=False)
            print(f"Column '{check_opjeck}' has been removed.")
        break
    # while True:
    #     column_to_drop = input("\nNhập tên thuộc tính bạn muốn xóa (hoặc nhập 'no' để không xóa): ")
    #     if column_to_drop.lower() == 'no':
    #         break
    #     elif column_to_drop in df.columns:
    #         df.drop(columns=[column_to_drop], inplace=True)
    #         df.to_csv('CC GENERAL.csv', index=False)
    #         print(f"Column '{column_to_drop}' has been removed.")
    #         break
    #     else:
    #         print(f"Column '{column_to_drop}' does not exist. Vui lòng nhập lại.")
    
    while True:
        choice = input("\nBạn có muốn xử lý giá trị khuyết bằng KNN Imputer? (yes/no): ")
        if choice.lower() == "yes":
            imputer = KNNImputer(n_neighbors=5)
            df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
            df_imputed.to_csv('CC GENERAL.csv', index=False)
            print("Missing values have been imputed using KNN Imputer.")
            break
        elif choice.lower() == "no":
            break
        else:
            print("Vui lòng nhập 'yes' hoặc 'no'.")
    
    input("\nPress Enter to return to the main menu...")
    intro()


def exploratory_data_analysis():
    data = pd.read_csv('CC GENERAL.csv')
    
    plt.figure(figsize=(10, 6))
    plt.scatter(data['TENURE'], data['BALANCE'], alpha=0.5, label='Balance')
    plt.scatter(data['TENURE'], data['CREDIT_LIMIT'], alpha=0.5, label='Credit Limit')
    plt.xlabel('Tenure (months)')
    plt.ylabel('Amount ($)')
    plt.title('Scatter Plot of Balance and Credit Limit by Tenure')
    plt.legend()
    plt.show()
    
    while True:
        choice = input("\nBạn có muốn thực hiện phân tích số lượng mua so với tổng giao dịch? (yes/no): ")
        if choice.lower() == "yes":
            analyze_purchases_vs_transactions(data)
            break
        elif choice.lower() == "no":
            break
        else:
            print("Vui lòng nhập 'yes' hoặc 'no'.")
    
    input("\nPress Enter to return to the main menu...")
    intro()

def analyze_purchases_vs_transactions(data):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    sns.scatterplot(ax=axes[0], x='PURCHASES', y='TENURE', data=data, label='Purchases', color='blue')
    axes[0].set_xlabel('Amount ($)')
    axes[0].set_ylabel('Tenure (months)')
    axes[0].set_title('Purchases by Tenure')
    axes[0].legend()

    sns.scatterplot(ax=axes[1], x='PURCHASES_TRX', y='TENURE', data=data, label='Total Transactions', color='red')
    axes[1].set_xlabel('Amount ($)')
    axes[1].set_ylabel('Tenure (months)')
    axes[1].set_title('Total Transactions by Tenure')
    axes[1].legend()

    plt.show()

    while True:
        choice = input("\nBạn có muốn phân tích sự tương quan của các thuộc tính? (yes/no): ")
        if choice.lower() == "yes":
            analyze_correlation(data)
            break
        elif choice.lower() == "no":
            break
        else:
            print("Vui lòng nhập 'yes' hoặc 'no'.")
    
    input("\nPress Enter to return to the main menu...")
    intro()

def analyze_correlation(data):
    correlation_matrix = data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Credit Card Data')
    plt.show()

    for col in correlation_matrix.columns:
        for row in correlation_matrix.index:
            if col != row:
                corr_value = correlation_matrix.loc[row, col]
                if -0.3 < corr_value < 0.3:
                    print(f"Độ tương quan giữa {row} và {col} không tốt: {corr_value:.2f}")
                elif -0.7 < corr_value < -0.3 or 0.7 > corr_value > 0.3:
                    print(f"Độ tương quan giữa {row} và {col} tương đối tốt: {corr_value:.2f}")
                elif -1 < corr_value < -0.7 or 1 > corr_value > 0.7:
                    print(f"Độ tương quan giữa {row} và {col} rất tốt: {corr_value:.2f}")

def data_normalization():
    data = pd.read_csv('CC GENERAL.csv')
    print("\nData before normalization:")
    print(data.head())

    # Chỉ lấy các thuộc tính liên quan đến hành vi sử dụng thẻ tín dụng
    features = data[['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE']]

    # Loại bỏ dữ liệu thiếu
    features.dropna(inplace=True)

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Chuyển đổi lại thành DataFrame và lưu vào file CSV
    features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)
    features_scaled_df.to_csv('CC GENERAL Normalized.csv', index=False)

    print("\nData after normalization:")
    print(features_scaled_df.head())

    while True:
        choice = input("\nBạn có muốn giảm chiều dữ liệu bằng PCA? (yes/no): ")
        if choice.lower() == "yes":
            pca = PCA(n_components=2)  # Giảm xuống 2 chiều để dễ hình dung
            features_pca = pca.fit_transform(features_scaled)
            features_pca_df = pd.DataFrame(features_pca, columns=['PC1', 'PC2'])
            features_pca_df.to_csv('CC GENERAL PCA.csv', index=False)
            print("\nData after PCA:")
            print(features_pca_df.head())
            break
        elif choice.lower() == "no":
            break
        else:
            print("Vui lòng nhập 'yes' hoặc 'no'.")
    
    input("\nPress Enter to return to the main menu...")
    intro()

def choose_optimal_clusters():
    data = pd.read_csv('CC GENERAL.csv')

    # Chỉ lấy các thuộc tính liên quan đến hành vi sử dụng thẻ tín dụng
    features = data[['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE']]

    # Loại bỏ dữ liệu thiếu
    features.dropna(inplace=True)

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Elbow method để tìm số cụm tối ưu
    wss = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(features_scaled)
        wss.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(K, wss, 'bx-')
    plt.axvline(x=4, color='r', linestyle='--')
    plt.text(4.2, wss[3], 'Elbow at k=4', color='red')
    plt.xlabel('Number of clusters')
    plt.ylabel('Within-cluster Sum of Squares')
    plt.title('Elbow Method For Optimal number of clusters')
    plt.show()

    # Tìm số cụm tối ưu
    optimal_clusters = None
    for i in range(1, len(wss) - 1):
        if wss[i] - wss[i+1] < wss[i-1] - wss[i]:
            optimal_clusters = i + 1
            break

    if optimal_clusters is None:
        optimal_clusters = len(K)

    print(f"\nSố cụm tối ưu (Elbow method): {optimal_clusters}")

    input("\nPress Enter to return to the main menu...")
    intro()

def main():
    print("K-Means Clustering")
    print("--------------------")

    # Đọc dữ liệu từ file CSV
    data = pd.read_csv('CC GENERAL PCA.csv')
    x = data 
    # Chỉ lấy các thuộc tính liên quan đến hành vi sử dụng thẻ tín dụng
    # features = data[['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE']]

    # # Loại bỏ dữ liệu thiếu
    # features.dropna(inplace=True)

    # # Chuẩn hóa dữ liệu
    # scaler = StandardScaler()
    # features_scaled = scaler.fit_transform(features)

    # Elbow method để tìm số cụm tối ưu
    wss = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(x)
        wss.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(K, wss, 'bx-')
    plt.axvline(x=4, color='r', linestyle='--')
    plt.text(4.2, wss[3], 'Elbow at k=4', color='red')
    plt.xlabel('Number of clusters')
    plt.ylabel('Within-cluster Sum of Squares')
    plt.title('Elbow Method For Optimal number of clusters')
    plt.show()

    # Tìm số cụm tối ưu
    # optimal_clusters = None
    # for i in range(1, len(wss) - 1):
    #     if wss[i] - wss[i+1] < wss[i-1] - wss[i]:
    #         optimal_clusters = i + 1
    #         break

    # if optimal_clusters is None:
    #     optimal_clusters = len(K)

    # print(f"\nSố cụm tối ưu (Elbow method): {optimal_clusters}")

    # Nhập số cụm từ người dùng
    # while True:
    #     try:
    #         num_clusters = int(input("Nhập số cụm (0 để sử dụng số cụm tối ưu từ Elbow method): "))
    #         if num_clusters < 0:
    #             print("Số cụm phải là số nguyên không âm.")
    #         elif num_clusters == 0:
    #             num_clusters = x
    #             break
    #         else:
    #             break
    #     except ValueError:
    #         print("Số cụm phải là số nguyên.")

    # # Sử dụng K-Means phân cụm với số cụm đã chọn
    # kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    # kmeans.fit(x)

    # # Thêm cột kết quả phân cụm vào DataFrame
    # x['Cluster'] = kmeans.labels_

    # # Hiển thị kết quả
    # print("\nKết quả phân cụm:")
    # print(x.head())

   # Nhập số cụm từ người dùng
    while True:
        try:
            num_clusters = int(input("Nhập số cụm (0 để sử dụng số cụm tối ưu từ Elbow method): "))
            if num_clusters < 0:
                print("Số cụm phải là số nguyên không âm.")
            elif num_clusters == 0:
                num_clusters = 4  # Giả sử số cụm tối ưu là 4 (từ Elbow method)
                break
            else:
                break
        except ValueError:
            print("Số cụm phải là số nguyên.")

    # Sử dụng K-Means phân cụm với số cụm đã chọn
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(x)

    # Thêm cột kết quả phân cụm vào DataFrame
    x['Cluster'] = kmeans.labels_

    # Hiển thị kết quả
    print("\nKết quả phân cụm:")
    print(x.head())

    # Yêu cầu người dùng nhập đường dẫn tới file PCA
    while True:
        pca_file_path = input("\nNhập đường dẫn tới file PCA (ví dụ: 'CC GENERAL PCA.csv'): ")
        try:
            # Đọc dữ liệu từ file PCA
            x = pd.read_csv(pca_file_path)
            print("\nDữ liệu PCA:")
            print(x.head())
            break
        except FileNotFoundError:
            print(f"File '{pca_file_path}' không tồn tại. Vui lòng nhập lại đường dẫn chính xác.")

    # Hỏi người dùng có muốn vẽ biểu đồ phân cụm không
    while True:
        choice = input("\nBạn có muốn vẽ biểu đồ phân cụm? (yes/no): ")
        if choice.lower() == "yes":
            # Áp dụng K-means trên dữ liệu PCA
            kmeans_pca = KMeans(n_clusters=num_clusters, random_state=42)
            kmeans_pca.fit(x)
            x['Cluster'] = kmeans_pca.labels_

            # Vẽ đồ thị phân cụm với PCA
            plt.figure(figsize=(10, 6))
            plt.scatter(x.iloc[:, 0], x.iloc[:, 1], c=x['Cluster'], cmap='viridis')
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
            plt.title(f'K-Means Clustering on PCA Data (k={num_clusters})')
            plt.colorbar(label='Cluster')
            plt.show()

            # Vẽ biểu đồ Silhouette với PCA
            silhouette_avg = silhouette_score(x, kmeans_pca.labels_)
            sample_silhouette_values = silhouette_samples(x, kmeans_pca.labels_)

            plt.figure(figsize=(10, 6))
            y_lower = 10
            for i in range(num_clusters):
                ith_cluster_silhouette_values = sample_silhouette_values[kmeans_pca.labels_ == i]
                ith_cluster_silhouette_values.sort()
                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                plt.fill_betweenx(np.arange(y_lower, y_upper),
                                  0, ith_cluster_silhouette_values,
                                  alpha=0.7)

                plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
                y_lower = y_upper + 10

            plt.title("Silhouette plot for the various clusters")
            plt.xlabel("Silhouette coefficient values")
            plt.ylabel("Cluster label")
            plt.axvline(x=silhouette_avg, color="red", linestyle="--")
            plt.show()

            # Vẽ biểu đồ Waffle với PCA
            try:
                cluster_counts = x['Cluster'].value_counts(normalize=True).sort_index() * 100
                plt.figure(
                    FigureClass=Waffle,
                    rows=5,
                    values=cluster_counts,
                    title={'label': 'Waffle Chart of Cluster Distribution (Percentage)', 'loc': 'center'},
                    labels=[f"Cluster {i} ({count:.1f}%)" for i, count in enumerate(cluster_counts)],
                    legend={'loc': 'upper left', 'bbox_to_anchor': (1, 1)}
                )
                plt.show()
            except ModuleNotFoundError:
                print("Module 'pywaffle' is not installed. Please install it to view the Waffle chart.")
























            break
        elif choice.lower() == "no":
            break
        else:
            print("Vui lòng nhập 'yes' hoặc 'no'.")
    
    # Quay lại menu chính
    intro()

if __name__ == "__main__":
    intro()