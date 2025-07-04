import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
import scipy.stats as stats
from scipy.stats import norm
from mpl_toolkits.basemap import Basemap
import json
import os
from sklearn.metrics import mean_squared_error, precision_recall_curve, average_precision_score
import matplotlib.colors as mcolors
import locale
import sklearn.metrics as metrics

# Set locale untuk format angka Indonesia
locale.setlocale(locale.LC_NUMERIC, 'id_ID.UTF-8')

def format_number(value):
    """Format angka sesuai dengan konvensi Indonesia (koma sebagai desimal, titik sebagai pemisah ribuan)."""
    return locale.format_string("%.2f", value, grouping=True).replace(",", "_").replace(".", ",").replace("_", ".")


def calibration_plot(y_pred, y_true, bins=100, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111)

    y_true = y_true.reshape(-1, 1)
    prob = np.sum(
        y_pred[:, :, 0] * (1 - norm.cdf((y_true - y_pred[:, :, 1]) / y_pred[:, :, 2])),
        axis=-1, keepdims=True
    )

    sns.histplot(prob, bins=bins, binrange=(0, 1), stat="density", ax=ax)  # Updated line
    #sns.distplot(prob, norm_hist=True, bins=bins, hist_kws={'range': (0, 1)}, kde=False, ax=ax)
    ax.axhline(1., linestyle='--', color='r')
    ax.set_xlim(0, 1)
    ax.set_ylim(0)
    
    return ax


# Plot function
def true_predicted(y_true, y_pred, agg='mean', quantile=True, ms=None, ax=None,
                   show_percentile_lines=True, bin_width=0.2, min_points=10):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    from sklearn.metrics import r2_score
    from matplotlib.lines import Line2D
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    ax.set_aspect('equal')

    if quantile:
        c_quantile = np.sum(
            y_pred[:, :, 0] * (1 - norm.cdf((y_true.reshape(-1, 1) - y_pred[:, :, 1]) / y_pred[:, :, 2])),
            axis=-1)
    else:
        c_quantile = np.full(len(y_true), 0.5)

    if agg == 'mean':
        y_pred_point = np.sum(y_pred[:, :, 0] * y_pred[:, :, 1], axis=1)
    elif agg == 'point':
        y_pred_point = y_pred
    else:
        raise ValueError(f'Aggregation type \"{agg}\" unknown')

    limits = (np.min(y_true) - 0.5, np.max(y_true) + 0.5)
    ax.plot(limits, limits, 'k-', zorder=1, label='Reference Line (y = x)', linewidth=1.5)

    scatter = ax.scatter(
        y_true, y_pred_point,
        c=c_quantile, cmap='coolwarm',
        edgecolors='black', zorder=2
    )
    #cbar = plt.colorbar(scatter, ax=ax)
    #cbar.set_label('Prediction Quantile', fontsize=24)
    #cbar.ax.tick_params(labelsize=24)

    if np.percentile(y_true, 95) <= 3.5 and np.percentile(y_true, 5) <= 0:
        ax.set_xlabel('True log(PGA) (m/s²)', fontsize=24)
        ax.set_ylabel('Predicted log(PGA) (m/s²)', fontsize=24)
        ax.tick_params(axis='both', labelsize=24)

    else:
        ax.set_xlabel('True Magnitude', fontsize=24)
        ax.set_ylabel('Predicted Magnitude', fontsize=24)
        ax.tick_params(axis='both', labelsize=24)        

    r2 = r2_score(y_true, y_pred_point)
    ax.text(limits[0], limits[1], f"$R^2={r2:.2f}$", va='top', fontsize=24)

    handles = []
    if show_percentile_lines:
        bins = np.arange(np.floor(y_true.min()), np.ceil(y_true.max()) + bin_width, bin_width)
        bin_centers, p5_list, p50_list, p95_list = [], [], [], []

        for i in range(len(bins) - 1):
            bin_mask = (y_true >= bins[i]) & (y_true < bins[i + 1])
            if np.sum(bin_mask) >= min_points:
                bin_center = 0.5 * (bins[i] + bins[i + 1])
                y_pred_bin = y_pred_point[bin_mask]
                bin_centers.append(bin_center)
                p5_list.append(np.percentile(y_pred_bin, 5))
                p50_list.append(np.percentile(y_pred_bin, 50))
                p95_list.append(np.percentile(y_pred_bin, 95))

        if len(bin_centers) > 0:
            ax.plot(bin_centers, p5_list, color='forestgreen', linestyle='--', linewidth=3)
            ax.plot(bin_centers, p50_list, color='forestgreen', linestyle='-', linewidth=3)
            ax.plot(bin_centers, p95_list, color='forestgreen', linestyle='--', linewidth=3)

            handles += [
                Line2D([0], [0], color='forestgreen', linestyle='-', linewidth=3,
                       label='Prediction Median'),
                Line2D([0], [0], color='forestgreen', linestyle='--', linewidth=3,
                       label='5th and 95th Percentiles')
            ]

    handles += [
        Line2D([0], [0], marker='o', color='w', label='Underestimate (quantile > 0.6)',
               markerfacecolor=plt.cm.coolwarm(0.9), markeredgecolor='black', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Accurate (quantile 0.4–0.6)',
               markerfacecolor=plt.cm.coolwarm(0.5), markeredgecolor='black', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Overestimate (quantile < 0.4)',
               markerfacecolor=plt.cm.coolwarm(0.1), markeredgecolor='black', markersize=10),
        Line2D([0], [0], color='k', lw=1.5, label='Reference Line (y = x)')
    ]

    #ax.legend(handles=handles, loc='lower right', fontsize=16)
    return ax, scatter





import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import locale


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def plot_pred_true_coordinates(pred_coord, true_coord, pos_offset=(0, 0), save_path=None):
    D2KM = 111.32  # Degrees to kilometers conversion factor

    # Copy pred_coord to avoid modifying the original data
    pred_coord_transformed = np.copy(pred_coord)

    # Convert only latitude and longitude (columns 0 and 1)
    pred_coord_transformed[:, 0] = pred_coord[:, 0] * 100 / D2KM + pos_offset[0]  # Latitude
    pred_coord_transformed[:, 1] = pred_coord[:, 1] * 100 / D2KM + pos_offset[1]  # Longitude
    pred_coord_transformed[:, 2] = pred_coord[:, 2] * 100  # Depth (km), no unit change

    # Compute RMSE
    rmse_lat = np.sqrt(mean_squared_error(true_coord[:, 0], pred_coord_transformed[:, 0]))
    rmse_lon = np.sqrt(mean_squared_error(true_coord[:, 1], pred_coord_transformed[:, 1]))
    rmse_depth = np.sqrt(mean_squared_error(true_coord[:, 2], pred_coord_transformed[:, 2]))

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    plt.rcParams.update({'font.size': 14, 'axes.grid': True, 'grid.alpha': 0.5})

    # Latitude plot
    axs[0].scatter(true_coord[:, 0], pred_coord_transformed[:, 0], alpha=0.6, edgecolors='black')
    axs[0].plot([true_coord[:, 0].min(), true_coord[:, 0].max()],
                [true_coord[:, 0].min(), true_coord[:, 0].max()], 'r--', label="y = x")
    axs[0].set_xlabel("True Latitude (°)")
    axs[0].set_ylabel("Predicted Latitude (°)")
    axs[0].set_title(f"Latitude Comparison\nRMSE: {rmse_lat:.2f}°")
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.6)

    # Longitude plot
    axs[1].scatter(true_coord[:, 1], pred_coord_transformed[:, 1], alpha=0.6, edgecolors='black')
    axs[1].plot([true_coord[:, 1].min(), true_coord[:, 1].max()],
                [true_coord[:, 1].min(), true_coord[:, 1].max()], 'r--', label="y = x")
    axs[1].set_xlabel("True Longitude (°)")
    axs[1].set_ylabel("Predicted Longitude (°)")
    axs[1].set_title(f"Longitude Comparison\nRMSE: {rmse_lon:.2f}°")
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.6)

    # Depth plot
    axs[2].scatter(true_coord[:, 2], pred_coord_transformed[:, 2], alpha=0.6, edgecolors='black')
    axs[2].plot([true_coord[:, 2].min(), true_coord[:, 2].max()],
                [true_coord[:, 2].min(), true_coord[:, 2].max()], 'r--', label="y = x")
    axs[2].set_xlabel("True Depth (km)")
    axs[2].set_ylabel("Predicted Depth (km)")
    axs[2].set_title(f"Depth Comparison\nRMSE: {rmse_depth:.2f} km")
    axs[2].legend()
    axs[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")


def plot_epicenter_comparison(pred_coord, true_coord, pos_offset=(0, 0), time=None, save_path=None):
    from mpl_toolkits.basemap import Basemap
    import matplotlib.pyplot as plt
    import numpy as np

    # Fungsi untuk menampilkan label koordinat dalam format LU, LS, BT, BB
    def format_latitude_label(lat):
        return f"{abs(lat)}° LS" if lat < 0 else f"{lat}° LU"

    def format_longitude_label(lon):
        return f"{abs(lon)}° BB" if lon < 0 else f"{lon}° BT"

def plot_epicenter_comparison(pred_coord, true_coord, pos_offset=(0, 0), time=None, save_path=None):
    from mpl_toolkits.basemap import Basemap
    import matplotlib.pyplot as plt
    import numpy as np

    # Fungsi untuk menampilkan label koordinat dalam format LU, LS, BT, BB
    def format_latitude_label(lat):
        return f"{abs(lat)}° LS" if lat < 0 else f"{lat}° LU"

    def format_longitude_label(lon):
        return f"{abs(lon)}° BB" if lon < 0 else f"{lon}° BT"

    D2KM = 111.32

    # Transformasi prediksi
    pred_coord *= 100
    pred_coord[:, :2] /= D2KM
    pred_coord[:, 0] += pos_offset[0]
    pred_coord[:, 1] += pos_offset[1]
    
# Expand longitude boundaries more significantly
    lon_min = min(pred_coord[:, 1].min(), true_coord[:, 0].min()) - 0.5  # Increased from 5 to 10
    lon_max = max(pred_coord[:, 1].max(), true_coord[:, 0].max()) + 0.5
    lat_min = min(pred_coord[:, 0].min(), true_coord[:, 1].min()) - 0.5
    lat_max = max(pred_coord[:, 0].max(), true_coord[:, 1].max()) + 0.5
    
    # Buat peta
    fig, ax = plt.subplots(figsize=(12, 8))
    m = Basemap(projection='merc',
                llcrnrlon=lon_min, llcrnrlat=lat_min, urcrnrlon=lon_max, urcrnrlat=lat_max,
                resolution='i', ax=ax)

    m.drawcoastlines()
    m.fillcontinents(color='white', lake_color='lightblue')
    m.drawmapboundary(fill_color='lightblue')
    #m.arcgisimage(service='World_Shaded_Relief', xpixels=1500, verbose=True)
    
    # Tambahkan garis lintang dan bujur (graticules)
    parallels = np.arange(int(lat_min), int(lat_max) + 1, 2.0)
    meridians = np.arange(int(lon_min), int(lon_max) + 1, 2.0)

    m.drawparallels(parallels, labels=[0, 0, 0, 0], fontsize=10, linewidth=0.5, dashes=[2, 2])
    m.drawmeridians(meridians, labels=[0, 0, 0, 0], fontsize=10, linewidth=0.5, dashes=[2, 2])

    # Tambahkan label lintang hanya dalam rentang yang sesuai
    for lat in parallels:
        if lat_min <= lat <= lat_max:  # Pastikan hanya dalam batas peta
            x, y = m(lon_min - 0.05, lat)  # Posisikan label sedikit ke kiri
            plt.text(x, y, format_latitude_label(lat), fontsize=10, ha='right', va='center', color='black')

    # Tambahkan label bujur hanya dalam rentang yang sesuai
    for lon in meridians:
        if  lon_min <= lon <= lon_max:  # Pastikan hanya dalam batas peta
            x, y = m(lon, lat_min - 0.05)  # Posisikan label sedikit ke bawah
            plt.text(x, y, format_longitude_label(lon), fontsize=10, ha='center', va='top', color='black')

    # Konversi koordinat ke sistem peta
    pred_x, pred_y = m(pred_coord[:, 1], pred_coord[:, 0])
    true_x, true_y = m(true_coord[:, 0], true_coord[:, 1])

    # Plot titik episenter sebenarnya dan prediksi
    ax.scatter(true_x, true_y, c='blue', label='True Location', alpha=0.7, s=40, edgecolors='k', zorder=5)
    ax.scatter(pred_x, pred_y, c='red', label='Predicted Location', alpha=0.7, s=40, edgecolors='k', zorder=5)

    # Tambahkan garis penghubung antar titik
    for px, py, tx, ty in zip(pred_x, pred_y, true_x, true_y):
        ax.plot([px, tx], [py, ty], 'gray', linestyle='--', linewidth=1.0, alpha=0.6)

    # Tambahkan waktu evaluasi di judul
    if time is not None:
        ax.set_title(f"Perbandingan Lokasi Episenter: Prediksi vs Sebenarnya\nWaktu = {time} s", fontsize=14)
    else:
        ax.set_title("Perbandingan Lokasi Episenter: Prediksi vs Sebenarnya", fontsize=14)

    ax.legend(loc='upper right')

    if save_path:
        plt.savefig(save_path, dpi=350, bbox_inches='tight')
        print(f"Plot saved at: {save_path}")
        
        


def plot_loc_rmse_vs_time(stats_json_path, save_path=None):
    """
    Membuat plot rata-rata RMSE untuk semua gempa berdasarkan variasi waktu.

    Parameters:
    - stats_json_path: str, path ke file 'stats.json' hasil evaluasi.
    - save_path: str, path untuk menyimpan plot (opsional).
    """
    # Load hasil evaluasi dari stats.json
    with open(stats_json_path, 'r') as f:
        stats = json.load(f)

    times = np.array(stats['times'])  # Waktu evaluasi
    loc_stats = np.array(stats['loc_stats'])  # Statistik lokasi: [rmse_hypo, mae_hypo, rmse_epi, mae_epi]

    # Ambil RMSE untuk hypocenter dan epicenter
    rmse_hypo = loc_stats[:, 0]  # Hypocenter RMSE
    rmse_epi = loc_stats[:, 2]   # Epicenter RMSE

    # Mengatur format lokal ke Indonesia
    locale.setlocale(locale.LC_NUMERIC, 'id_ID.UTF-8')

    # Buat plot
    plt.figure(figsize=(12, 6))
    plt.plot(times, rmse_hypo, label='RMSE Hiposenter', marker='o', color='blue')
    plt.plot(times, rmse_epi, label='RMSE Episenter', marker='s', color='green')
    plt.xlabel('Waktu (detik)')
    plt.ylabel('RMSE (km)')
    plt.title('Hubungan RMSE dengan Waktu')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    
    # Menggunakan format angka Indonesia
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: locale.format_string('%.0f', x, grouping=True)))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: locale.format_string('%.2f', y, grouping=True)))
    
    # Simpan plot jika path diberikan
    if save_path:
        plt.savefig(save_path, dpi=350, bbox_inches='tight')
        print(f"Plot disimpan di: {save_path}")
    else:
        plt.show()



def plot_loc_mae_vs_time(stats_json_path, save_path=None):
    """
    Membuat plot rata-rata MAE untuk semua gempa berdasarkan variasi waktu.

    Parameters:
    - stats_json_path: str, path ke file 'stats.json' hasil evaluasi.
    - save_path: str, path untuk menyimpan plot (opsional).
    """
    # Load hasil evaluasi dari stats.json
    with open(stats_json_path, 'r') as f:
        stats = json.load(f)

    # Ambil waktu evaluasi dan statistik lokasi
    times = np.array(stats['times'])  # Waktu evaluasi
    loc_stats = np.array(stats['loc_stats'])  # Statistik lokasi: [rmse_hypo, mae_hypo, rmse_epi, mae_epi]

    # Ambil MAE untuk hypocenter dan epicenter
    mae_hypo = loc_stats[:, 1]  # Hypocenter MAE
    mae_epi = loc_stats[:, 3]   # Epicenter MAE

    # Mengatur format lokal ke Indonesia
    locale.setlocale(locale.LC_NUMERIC, 'id_ID.UTF-8')

    # Buat plot
    plt.figure(figsize=(12, 6))
    plt.plot(times, mae_hypo, label='MAE Hiposenter', marker='o', color='blue')
    plt.plot(times, mae_epi, label='MAE Episenter', marker='s', color='green')
    plt.xlabel('Waktu (detik)')
    plt.ylabel('MAE (km)')
    plt.title('MAE Hiposenter dan Episenter')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    
    # Menggunakan format angka Indonesia
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: locale.format_string('%.0f', x, grouping=True)))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: locale.format_string('%.2f', y, grouping=True)))
    
    # Simpan plot jika path diberikan
    if save_path:
        plt.savefig(save_path, dpi=350, bbox_inches='tight')
        print(f"Plot disimpan di: {save_path}")
\



def plot_rmse_pga_vs_time(times, rmse_pga_list, save_path=None):
    """
    Membuat plot rata-rata RMSE PGA terhadap waktu.

    Parameters:
    - times: list atau np.ndarray, waktu evaluasi (dalam detik).
    - rmse_pga_list: list atau np.ndarray, nilai RMSE PGA pada setiap waktu evaluasi.
    - save_path: str, jalur untuk menyimpan plot (opsional).
    """
    plt.figure(figsize=(12, 7))
    plt.plot(times, rmse_pga_list, marker='o', linestyle='-', color='blue', label='RMSE PGA')
    plt.xlabel('Time (s)')
    plt.ylabel('RMSE PGA')
    plt.title('RMSE PGA vs Time')
    plt.grid(True)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')



def plot_mae_pga_vs_time(times, mae_pga_list, save_path=None):
    """
    Membuat plot rata-rata MAE PGA terhadap waktu.

    Parameters:
    - times: list atau np.ndarray, waktu evaluasi (dalam detik).
    - mae_pga_list: list atau np.ndarray, nilai MAE PGA pada setiap waktu evaluasi.
    - save_path: str, jalur untuk menyimpan plot (opsional).
    """
    plt.figure(figsize=(12, 7))
    plt.plot(times, mae_pga_list, marker='o', linestyle='-', color='green', label='MAE PGA')
    plt.xlabel('Time (s)')
    plt.ylabel('MAE PGA')
    plt.title('MAE PGA vs Time')
    plt.grid(True)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')


import numpy as np
import matplotlib.pyplot as plt

def plot_warning_times(warning_time_information, alpha, times_pga, save_path=None):
    """
    Plots warning times as a function of distance for different alpha levels.

    Parameters:
    - warning_time_information: List of tuples containing (pga_times_pred, pga_times_true, dist).
    - alpha: List of alpha values for probability thresholds.
    - times_pga: List of times at which PGA thresholds are evaluated.
    - save_path: Optional path to save the plot.
    """
    plt.figure(figsize=(10, 6))

    for i, prob_threshold in enumerate(alpha):
        warning_times = []
        distances = []

        for event_info in warning_time_information:
            pga_times_pred, pga_times_true, dist = event_info

            # 🔍 Debugging: Periksa bentuk awal array
            print(f"\nBefore Fix: Shape of pga_times_true: {pga_times_true.shape}, Shape of pga_times_pred: {pga_times_pred.shape}")

            # 🛠️ **Solusi: Pilih Dimensi yang Benar**
            if len(pga_times_pred.shape) == 3:
                print(f"Selecting pga_times_pred[:, :, {i}] to match pga_times_true")
                pga_times_pred = pga_times_pred[:, :, i]  # Pilih dimensi yang sesuai

            # 🔍 Debugging: Periksa bentuk setelah pemilihan dimensi
            print(f"After Fix: Shape of pga_times_true: {pga_times_true.shape}, Shape of pga_times_pred: {pga_times_pred.shape}")

            # 🚀 **Lakukan operasi pengurangan setelah dimensi sesuai**
            warning_time = pga_times_true - pga_times_pred

            # Simpan hasil untuk plotting
            warning_times.extend(warning_time.flatten())
            distances.extend(dist.flatten())

        # Konversi ke array NumPy
        warning_times = np.array(warning_times)
        distances = np.array(distances)

        # 🛠️ **Debugging jumlah elemen sebelum filtering**
        print(f"Before filtering: warning_times.shape = {warning_times.shape}, distances.shape = {distances.shape}")

        # 🛠️ **Pastikan jumlah elemen sama sebelum filtering**
        min_length = min(len(warning_times), len(distances))
        warning_times = warning_times[:min_length]
        distances = distances[:min_length]

        # 🛠️ **Filter nilai NaN sebelum plotting**
        valid_mask = ~np.isnan(warning_times)
        warning_times = warning_times[valid_mask]
        distances = distances[valid_mask[:len(distances)]]  # Pastikan valid_mask memiliki panjang yang sama dengan distances

        # 🔹 **Plot hasil per alpha**
        plt.scatter(distances, warning_times, label=f'α = {str(prob_threshold).replace(".", ",")}', alpha=0.6)

    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    
    # 🛠️ **Perbesar ukuran font pada sumbu X dan Y**
    plt.xlabel('Distance (km)', fontsize=14, fontweight='bold')
    plt.ylabel('Warning Time (s)', fontsize=14, fontweight='bold')
    
    plt.title('Warning Time vs Distance for Different Alpha Levels', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True)

    # 🛠️ **Simpan Gambar jika `save_path` diberikan**
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Warning times plot saved at: {save_path}")  




def plot_precision_recall_pga(pga_pred, pga_true, thresholds=[0.01, 0.02, 0.05, 0.1, 0.2], save_path=None):
    """
    Plots Precision vs Recall for various PGA thresholds using a calculation method
    similar to calc_pga_stats.

    Parameters:
    - pga_pred: list or np.ndarray, predicted PGA values (expected shape: (N, M, 2) or similar).
    - pga_true: list or np.ndarray, true PGA values.
    - thresholds: list, PGA thresholds for precision-recall curves.
    - save_path: str, optional path to save the plot.
    """

    # 🛠️ *Step 1: Pastikan pga_pred memiliki bentuk yang seragam*
    pga_pred = [np.array(x) for x in pga_pred if np.shape(x)]  # Hapus elemen kosong
    if len(pga_pred) > 0:
        pga_pred = np.concatenate(pga_pred, axis=0)  # Gabungkan array
    else:
        print("WARNING: pga_pred is empty after filtering!")
        return

    # 🛠️ *Step 2: Pastikan pga_true juga memiliki bentuk yang seragam*
    pga_true = [np.array(x) for x in pga_true if np.shape(x)]  # Hapus elemen kosong
    if len(pga_true) > 0:
        pga_true = np.concatenate(pga_true, axis=0)  # Gabungkan array
    else:
        print("WARNING: pga_true is empty after filtering!")
        return

    # Debugging: Cek bentuk array setelah diproses
    print(f"Shape of pga_pred after processing: {pga_pred.shape}")
    print(f"Shape of pga_true after processing: {pga_true.shape}")

    # 🛠️ *Step 3: Perhitungan Mean PGA seperti calc_pga_stats*
    if len(pga_pred.shape) == 3 and pga_pred.shape[-1] == 3:
        mean_pga_pred = np.sum(pga_pred[:, :, 0] * pga_pred[:, :, 1], axis=1)
    else:
        mean_pga_pred = pga_pred  # Jika tidak 3D, gunakan langsung

    # Masking untuk menghindari NaN dan inf
    mask = ~np.logical_or(np.isnan(pga_true), np.isinf(pga_true))
    pga_true = pga_true[mask]
    mean_pga_pred = mean_pga_pred[mask]

    if len(pga_true) == 0 or len(mean_pga_pred) == 0:
        print("WARNING: No valid PGA values after filtering NaN and Inf.")
        return

    # 🛠️ *Step 4: Plot Precision-Recall Curve*
    fig, axs = plt.subplots(1, len(thresholds), figsize=(20, 5), sharey=True)

    for idx, threshold in enumerate(thresholds):
        ax = axs[idx]

        # Konversi PGA ke nilai biner berdasarkan threshold
        true_binary = (pga_true >= threshold).astype(int)

        if len(mean_pga_pred) == 0 or len(true_binary) == 0:
            print(f"WARNING: No valid PGA values for threshold {threshold}")
            continue

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(true_binary.ravel(), mean_pga_pred.ravel())
        avg_precision = average_precision_score(true_binary.ravel(), mean_pga_pred.ravel())

        # Plot curve
        ax.plot(recall, precision, label=f'AP={avg_precision:.2f}', color='blue', marker='s')

        # Random baseline
        ax.plot([0, 1], [0, 1], 'k--', lw=1)

        ax.text(0.05, 0.05, f'{avg_precision:.2f}', fontsize=12, color='blue', transform=ax.transAxes)

        ax.set_title(f'{int(threshold * 100)}%g')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.grid(True)

    axs[0].set_ylabel('Precision')
    for ax in axs:
        ax.set_xlabel('Recall')

    plt.suptitle('Precision-Recall Curve for PGA Thresholds', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 🛠️ *Step 5: Simpan Gambar jika diperlukan*
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Precision-Recall plot saved at: {save_path}")

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

def plot_precision_recall_curve(pga_pred, pga_true, thresholds, alpha_values, save_path=None):
    """
    Memplot Precision-Recall Curve sebagai subplot untuk berbagai threshold PGA.
    
    Parameters:
    - pga_pred: list or np.ndarray, predicted PGA values (expected shape: (N, M, 3) or similar).
    - pga_true: list or np.ndarray, true PGA values.
    - thresholds: list, nilai PGA thresholds untuk ditampilkan di subplot.
    - alpha_values: list, nilai alpha yang digunakan dalam perhitungan.
    - save_path: str, optional path untuk menyimpan plot sebagai file.
    """

    # 🛠️ **Step 1: Pastikan `pga_pred` dan `pga_true` memiliki bentuk yang seragam**
    pga_pred = [np.array(x) for x in pga_pred if np.shape(x)]  # Hapus elemen kosong
    if len(pga_pred) > 0:
        pga_pred = np.concatenate(pga_pred, axis=0)  # Gabungkan array
    else:
        print("WARNING: pga_pred is empty after filtering!")
        return

    pga_true = [np.array(x) for x in pga_true if np.shape(x)]  # Hapus elemen kosong
    if len(pga_true) > 0:
        pga_true = np.concatenate(pga_true, axis=0)  # Gabungkan array
    else:
        print("WARNING: pga_true is empty after filtering!")
        return

    # Debugging: Cek bentuk array setelah diproses
    print(f"Shape of pga_pred after processing: {pga_pred.shape}")
    print(f"Shape of pga_true after processing: {pga_true.shape}")

    # 🛠️ **Step 2: Perhitungan Mean PGA seperti `calc_pga_stats`**
    if len(pga_pred.shape) == 3 and pga_pred.shape[-1] == 3:
        mean_pga_pred = np.sum(pga_pred[:, :, 0] * pga_pred[:, :, 1], axis=1)
    else:
        mean_pga_pred = pga_pred  # Jika tidak 3D, gunakan langsung

    # Masking untuk menghindari NaN dan inf
    mask = ~np.logical_or(np.isnan(pga_true), np.isinf(pga_true))
    pga_true = pga_true[mask]
    mean_pga_pred = mean_pga_pred[mask]

    if len(pga_true) == 0 or len(mean_pga_pred) == 0:
        print("WARNING: No valid PGA values after filtering NaN and Inf.")
        return

    # 🛠️ **Step 3: Buat subplot untuk setiap threshold**
    num_thresholds = len(thresholds)
    fig, axes = plt.subplots(1, num_thresholds, figsize=(5 * num_thresholds, 5), sharey=True)

    if num_thresholds == 1:
        axes = [axes]  # Jika hanya satu threshold, jadikan dalam bentuk list agar tetap iteratif

    # Loop melalui setiap threshold dan buat subplot
    for idx, threshold in enumerate(thresholds):
        ax = axes[idx]

        # Konversi `pga_true` ke nilai biner untuk threshold ini
        y_true = (pga_true >= threshold).astype(int)

        # Pastikan `y_true` memiliki kedua kelas (0 dan 1) sebelum menghitung PR curve
        if len(np.unique(y_true)) < 2:
            print(f"WARNING: y_true untuk threshold {threshold} hanya memiliki satu kelas. Melewati perhitungan.")
            continue

        auc_scores = []
        for alpha in alpha_values:
            # Precision-Recall Curve
            precision, recall, _ = metrics.precision_recall_curve(y_true, mean_pga_pred)
            auc_pr = metrics.average_precision_score(y_true, mean_pga_pred)
            auc_scores.append(auc_pr)

            # Plot PR Curve
            ax.plot(recall, precision, label=f"α={alpha:.2f} (AUC={auc_pr:.2f})")

        # Menemukan titik F1-score tertinggi
        f1_scores = 2 * (np.array(precision) * np.array(recall)) / (np.array(precision) + np.array(recall) + 1e-9)
        best_idx = np.argmax(f1_scores)
        ax.scatter(recall[best_idx], precision[best_idx], s=100, color='red', label='Best F1')

        # Format subplot
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"PR Curve for Threshold {threshold:.2f}")
        ax.legend()
        ax.grid(True)

    # Atur tata letak subplot
    plt.tight_layout()

    # 🛠️ **Step 4: Simpan Gambar jika `save_path` diberikan**
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Precision-Recall Curve plot saved at: {save_path}")



def plot_precision_recall_f1(pga_pred, pga_true, thresholds, save_path=None):
    """
    Memplot Precision, Recall, dan F1-score untuk berbagai threshold PGA dan menyimpannya sebagai gambar terpisah.
    
    Parameters:
    - pga_pred: list or np.ndarray, predicted PGA values (expected shape: (N, M, 3) or similar).
    - pga_true: list or np.ndarray, true PGA values.
    - thresholds: list, nilai PGA thresholds untuk ditampilkan di sumbu atas.
    - save_path: str, optional path untuk menyimpan plot sebagai file.
    """

    # 🛠️ **Step 1: Pastikan `pga_pred` dan `pga_true` memiliki bentuk yang seragam**
    pga_pred = [np.array(x) for x in pga_pred if np.shape(x)]  # Hapus elemen kosong
    if len(pga_pred) > 0:
        pga_pred = np.concatenate(pga_pred, axis=0)  # Gabungkan array
    else:
        print("WARNING: pga_pred is empty after filtering!")
        return

    pga_true = [np.array(x) for x in pga_true if np.shape(x)]  # Hapus elemen kosong
    if len(pga_true) > 0:
        pga_true = np.concatenate(pga_true, axis=0)  # Gabungkan array
    else:
        print("WARNING: pga_true is empty after filtering!")
        return

    # Debugging: Cek bentuk array setelah diproses
    print(f"Shape of pga_pred after processing: {pga_pred.shape}")
    print(f"Shape of pga_true after processing: {pga_true.shape}")

    # 🛠️ **Step 2: Perhitungan Mean PGA seperti `calc_pga_stats`**
    if len(pga_pred.shape) == 3 and pga_pred.shape[-1] == 3:
        mean_pga_pred = np.sum(pga_pred[:, :, 0] * pga_pred[:, :, 1], axis=1)
    else:
        mean_pga_pred = pga_pred  # Jika tidak 3D, gunakan langsung

    # Masking untuk menghindari NaN dan inf
    mask = ~np.logical_or(np.isnan(pga_true), np.isinf(pga_true))
    pga_true = pga_true[mask]
    mean_pga_pred = mean_pga_pred[mask]

    if len(pga_true) == 0 or len(mean_pga_pred) == 0:
        print("WARNING: No valid PGA values after filtering NaN and Inf.")
        return

    # 🛠️ **Step 3: Hitung Precision, Recall, dan F1-score untuk berbagai threshold**
    precision_list = []
    recall_list = []
    f1_list = []

    for threshold in thresholds:
        y_true = (pga_true >= threshold).astype(int)
        y_pred = (mean_pga_pred >= threshold).astype(int)

        # Cek apakah ada kedua kelas (0 dan 1) di `y_true`
        if len(np.unique(y_true)) < 2:
            print(f"WARNING: y_true untuk threshold {threshold} hanya memiliki satu kelas. Melewati perhitungan.")
            precision_list.append(0.0)
            recall_list.append(0.0)
            f1_list.append(0.0)
            continue

        # Hitung Precision, Recall, dan F1-score
        precision = metrics.precision_score(y_true, y_pred, zero_division=0)
        recall = metrics.recall_score(y_true, y_pred, zero_division=0)
        f1 = metrics.f1_score(y_true, y_pred, zero_division=0)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

        print(f"Threshold {threshold:.2f}: Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}")

    # 🛠️ **Step 4: Buat 3 Subplot untuk Precision, Recall, dan F1-score**
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Subplot Precision
    axes[0].plot(thresholds, precision_list, label="Precision", marker="o", linestyle="-", color="blue")
    axes[0].set_xlabel("PGA Threshold")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("Precision vs PGA Threshold")
    axes[0].grid(True)
    axes[0].legend()

    # Subplot Recall
    axes[1].plot(thresholds, recall_list, label="Recall", marker="s", linestyle="-", color="green")
    axes[1].set_xlabel("PGA Threshold")
    axes[1].set_ylabel("Recall")
    axes[1].set_title("Recall vs PGA Threshold")
    axes[1].grid(True)
    axes[1].legend()

    # Subplot F1-score
    axes[2].plot(thresholds, f1_list, label="F1-score", marker="d", linestyle="-", color="red")
    axes[2].set_xlabel("PGA Threshold")
    axes[2].set_ylabel("F1-score")
    axes[2].set_title("F1-score vs PGA Threshold")
    axes[2].grid(True)
    axes[2].legend()

    # Atur tata letak subplot
    plt.tight_layout()

    # 🛠️ **Step 5: Simpan Gambar jika `save_path` diberikan**
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Precision-Recall-F1 plot saved at: {save_path}")  

  
    

def plot_event_trace_pga(pga_thresholds, train_events, test_events, train_traces, test_traces):
    """
    Plot number of events and traces exceeding each PGA threshold for training and test sets.
    
    Parameters:
    - pga_thresholds: list or array, PGA thresholds in percentage g
    - train_events: list or array, number of events exceeding thresholds in training set
    - test_events: list or array, number of events exceeding thresholds in test set
    - train_traces: list or array, number of traces exceeding thresholds in training set
    - test_traces: list or array, number of traces exceeding thresholds in test set
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    
    # Define plotting properties
    plot_settings = {
        'marker': 'o', 'linestyle': '-', 'linewidth': 2, 'markersize': 6
    }
    
    # Plot Events
    axes[0].plot(pga_thresholds, train_events, color='gray', label='Train', **plot_settings)
    axes[0].plot(pga_thresholds, test_events, color='black', label='Test', **plot_settings)
    axes[0].set_title("Events")
    
    # Plot Traces
    axes[1].plot(pga_thresholds, train_traces, color='gray', label='Train', **plot_settings)
    axes[1].plot(pga_thresholds, test_traces, color='black', label='Test', **plot_settings)
    axes[1].set_title("Traces")
    
    # Shared axis properties
    for ax in axes:
        ax.set_yscale("log")  # Log scale for better visibility
        ax.set_xticks(pga_thresholds)
        ax.set_xticklabels([f"{t}%" for t in pga_thresholds], rotation=45)
        ax.set_xlabel("PGA threshold [g]")
    
    axes[0].set_ylabel("Count (log scale)")
    
    # Add legend
    axes[1].legend(loc='lower right', frameon=True, title="Dataset")
    
    plt.tight_layout()
    return fig


def plot_auc_warning_time(pga_thresholds, warning_times, auc_values):
    """
    Plot area under the precision-recall curve for different minimum warning times.
    
    Parameters:
    - pga_thresholds: list or array, PGA thresholds in percentage g
    - warning_times: list or array, different minimum warning times (seconds)
    - auc_values: 2D array (len(pga_thresholds) x len(warning_times)),
                  AUC values for each threshold and warning time
    """
    fig, axes = plt.subplots(1, len(pga_thresholds), figsize=(15, 4), sharey=True)
    
    for ax, pga, auc in zip(axes, pga_thresholds, auc_values):
        ax.plot(warning_times, auc, linestyle='-', linewidth=2, color='blue')
        ax.set_title(f"{pga}%g")
        ax.set_xlabel("Minimum warning time [s]")
        if ax == axes[0]:
            ax.set_ylabel("Area under curve")
        ax.set_xlim(min(warning_times), max(warning_times))
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
    return fig


def plot_mag_mae_vs_time(stats_json_path, save_path=None):
    """
    Membuat plot rata-rata MAE magnitudo terhadap waktu.

    Parameters:
    - stats_json_path: str, path ke file 'stats.json' hasil evaluasi.
    - save_path: str, path untuk menyimpan plot (opsional).
    """
    # Load hasil evaluasi dari stats.json
    with open(stats_json_path, 'r') as f:
        stats = json.load(f)

    # Ambil waktu evaluasi dan statistik magnitudo
    times = np.array(stats['times'])  # Waktu evaluasi
    mag_stats = np.array(stats['mag_stats'])  # Statistik magnitudo: [r2, rmse, mae]

    # Ambil MAE dari magnitudo
    mae_mag = mag_stats[:, 2]  # Indeks 2 adalah MAE

    # Buat plot
    plt.figure(figsize=(10, 6))
    plt.plot(times, mae_mag, marker='s', linestyle='-', color='g', label='MAE Magnitudo')
    plt.xlabel('Time (s)')
    plt.ylabel('MAE')
    plt.title('MAE Magnitudo vs Waktu')
    plt.legend()
    plt.grid(True)

    # Simpan plot jika path diberikan
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved at: {save_path}")


def plot_mag_rmse_vs_time(stats_json_path, save_path=None):
    """
    Membuat plot rata-rata RMSE magnitudoTraceback (most recent call last):
  File "/home/team/TEAM-main/evaluate.py", line 538, in <module>
    plots.plot_warning_times(warning_time_information, alpha, times_pga, save_path=save_warning_plot_path)
  File "/home/team/TEAM-main/plots.py", line 313, in plot_warning_times
    warning_time = pga_times_true - pga_times_pred[:, i]
ValueError: operands could not be broadcast together with shapes (47,5) (47,7)  terhadap waktu.

    Parameters:
    - stats_json_path: str, path ke file 'stats.json' hasil evaluasi.
    - save_path: str, path untuk menyimpan plot (opsional).
    """
    # Load hasil evaluasi dari stats.json
    with open(stats_json_path, 'r') as f:
        stats = json.load(f)

    # Ambil waktu evaluasi dan statistik magnitudo
    times = np.array(stats['times'])  # Waktu evaluasi
    mag_stats = np.array(stats['mag_stats'])  # Statistik magnitudo: [r2, rmse, mae]

    # Ambil RMSE dari magnitudo
    rmse_mag = mag_stats[:, 1]  # Indeks 1 adalah RMSE

    # Buat plot
    plt.figure(figsize=(10, 6))
    plt.plot(times, rmse_mag, marker='o', linestyle='-', color='b', label='RMSE Magnitudo')
    plt.xlabel('Time (s)')
    plt.ylabel('RMSE')
    plt.title('RMSE Magnitudo vs Waktu')
    plt.legend()
    plt.grid(True)

    # Simpan plot jika path diberikan
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved at: {save_path}")


def plot_warning_time_histogram(warning_times, save_path=None):
    """
    Membuat histogram waktu peringatan (warning time).
    
    Parameters:
    - warning_times: np.ndarray atau list, waktu peringatan dalam detik.
    - save_path: str, path untuk menyimpan gambar (opsional).
    """
    plt.figure(figsize=(6, 5))
    
    # Buat histogram
    plt.hist(warning_times, bins=np.arange(0, max(warning_times) + 5, 5), 
             color='green', edgecolor='black', histtype='step', linewidth=2)

    # Atur label dan judul
    plt.xlabel("Warning time [s]", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xlim(0, max(warning_times) + 10)
    plt.ylim(0, None)
    plt.grid(True, linestyle="--", alpha=0.7)

    # Simpan atau tampilkan gambar
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Histogram disimpan di {save_path}")
    

def plot_tp_fp_fn_bar(model_names, tp_values, fn_values, fp_values, save_path=None):
    """
    Membuat stacked bar chart untuk TP, FN, dan FP.

    Parameters:
    - model_names: list, nama model atau metode yang dievaluasi.
    - tp_values: list atau array, jumlah True Positives.
    - fn_values: list atau array, jumlah False Negatives.
    - fp_values: list atau array, jumlah False Positives.
    - save_path: str, path untuk menyimpan gambar (opsional).
    """
    x = np.arange(len(model_names))  # Lokasi pada sumbu x

    # Buat bar chart dengan stacking
    fig, ax = plt.subplots(figsize=(6, 5))
    
    ax.bar(x, tp_values, label="TP", color="green")
    ax.bar(x, fn_values, bottom=tp_values, label="FN", color="red")
    ax.bar(x, fp_values, bottom=np.array(tp_values) + np.array(fn_values), label="FP", color="tan")

    # Tambahkan label
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.set_ylabel("Count")
    ax.legend()

    # Simpan gambar jika path diberikan
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot disimpan di {save_path}")


import numpy as np
import matplotlib.pyplot as plt

def plot_warning_time_vs_distance(pga_thresholds, distances, warning_times, save_path=None):
    """
    Plots warning time vs distance for different PGA thresholds.

    Parameters:
    - pga_thresholds: List of PGA thresholds.
    - distances: List or np.ndarray of distances.
    - warning_times: List or np.ndarray of warning times.
    - save_path: Optional path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # 🛠️ Debugging: Periksa tipe data sebelum diproses
    print(f"Type of warning_times: {type(warning_times)}")
    print(f"Type of distances: {type(distances)}")
    print(f"Warning times contains {len(warning_times)} elements.")

    # 🛠️ **Periksa apakah warning_times adalah list dari array dengan panjang berbeda**
    if isinstance(warning_times, list):
        if all(isinstance(wt, (list, np.ndarray)) for wt in warning_times):
            print("Flattening warning_times to uniform shape.")

            try:
                # 🔹 Gunakan np.concatenate() jika memungkinkan
                warning_times = np.concatenate([np.ravel(np.array(wt)) for wt in warning_times if wt is not None])
            except ValueError:
                print("WARNING: np.concatenate failed due to incompatible shapes. Using np.hstack instead.")
                warning_times = np.hstack([np.ravel(np.array(wt)) for wt in warning_times if wt is not None])
        else:
            # Jika ada elemen yang bukan list/array, ubah ke array NumPy langsung
            print("Converting warning_times to numpy array directly.")
            warning_times = np.array(warning_times, dtype=float)

    # 🔍 Debugging setelah konversi
    print(f"After Fix: distances.shape = {np.shape(distances)}, warning_times.shape = {np.shape(warning_times)}")

    # Pastikan panjang `distances` dan `warning_times` sama sebelum plotting
    min_length = min(len(distances), len(warning_times))
    distances = distances[:min_length]
    warning_times = warning_times[:min_length]

    # 🔹 Plot hasil
    ax.scatter(distances, warning_times, alpha=0.5, s=5)
    
    ax.set_xlabel('Distance (km)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Warning Time (s)', fontsize=14, fontweight='bold')
    ax.set_title('Warning Time vs Distance for Different PGA Thresholds', fontsize=16, fontweight='bold')
    ax.grid(True)

    # 🛠️ **Simpan gambar jika `save_path` diberikan**
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Warning time vs distance plot saved at: {save_path}")






def running_mean_std(x, y, bins=20):
    """
    Menghitung rata-rata dan standar deviasi residual PGA dalam bins tertentu.
    
    Parameters:
    - x: array-like, sumbu X (jarak episentral atau magnitudo).
    - y: array-like, sumbu Y (residual PGA).
    - bins: int, jumlah pembagian sumbu X.

    Returns:
    - bin_centers: array, titik tengah setiap bin.
    - running_mean: array, nilai rata-rata residual di setiap bin.
    - running_std: array, nilai standar deviasi residual di setiap bin.
    """
    bin_edges = np.logspace(np.log10(min(x)), np.log10(max(x)), bins) if min(x) > 0 else np.linspace(min(x), max(x), bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    running_mean = np.zeros(len(bin_centers))
    running_std = np.zeros(len(bin_centers))

    for i in range(len(bin_centers)):
        mask = (x >= bin_edges[i]) & (x < bin_edges[i+1])
        if np.sum(mask) > 5:  # Minimal 5 data agar running mean/STD valid
            running_mean[i] = np.mean(y[mask])
            running_std[i] = np.std(y[mask])
        else:
            running_mean[i] = np.nan
            running_std[i] = np.nan

    return bin_centers, running_mean, running_std

def plot_pga_residuals(epi_distances, magnitudes, pga_residuals, save_path=None):
    """
    Membuat plot residual PGA terhadap jarak episentral dan magnitudo.

    Parameters:
    - epi_distances: np.ndarray, jarak episentral (km).
    - magnitudes: np.ndarray, magnitudo gempa.
    - pga_residuals: np.ndarray, residual PGA (logaritmik).
    - save_path: str, path untuk menyimpan gambar (opsional).
    """
    fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharey=True)

    # Plot Residual vs Distance
    ax1 = axes[0]
    ax1.scatter(epi_distances, pga_residuals, alpha=0.2, s=5, color="blue")
    bin_centers, running_mean, running_std = running_mean_std(epi_distances, pga_residuals, bins=15)
    
    ax1.plot(bin_centers, running_mean, color="orange", linewidth=2)
    ax1.plot(bin_centers, running_mean + running_std, color="orange", linestyle="dashed", linewidth=2)
    ax1.plot(bin_centers, running_mean - running_std, color="orange", linestyle="dashed", linewidth=2)

    ax1.set_xscale("log")
    ax1.set_xlabel("Epicentral distance [km]")
    ax1.set_ylabel("$PGA_{pred} - PGA_{true}$")
    
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Plot Residual vs Magnitude
    ax2 = axes[1]
    ax2.scatter(magnitudes, pga_residuals, alpha=0.2, s=5, color="blue")
    bin_centers, running_mean, running_std = running_mean_std(magnitudes, pga_residuals, bins=10)
    
    ax2.plot(bin_centers, running_mean, color="orange", linewidth=2)
    ax2.plot(bin_centers, running_mean + running_std, color="orange", linestyle="dashed", linewidth=2)
    ax2.plot(bin_centers, running_mean - running_std, color="orange", linestyle="dashed", linewidth=2)
    
    # Tambahkan titik silang (+) untuk magnitudo dengan data terbatas
    for i, (m, mean, std) in enumerate(zip(bin_centers, running_mean, running_std)):
        if np.isnan(mean):
            continue
        ax2.errorbar(m, mean, yerr=std, fmt='o', color="orange", capsize=4, elinewidth=2)

    ax2.set_xlabel("Magnitude")
    ax2.set_ylabel("$PGA_{pred} - PGA_{true}$")

    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    
    # Simpan gambar jika path diberikan
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot disimpan di {save_path}")


def plot_calibration_diagrams(confidence_bins, accuracy_values, times, pga_thresholds, save_path=None):
    """
    Membuat calibration diagrams untuk satu metode (TEAM).
    
    Parameters:
    - confidence_bins: np.ndarray, nilai confidence yang dikelompokkan dalam bin.
    - accuracy_values: np.ndarray, akurasi aktual untuk setiap confidence bin.
                       Harus berbentuk (len(pga_thresholds), len(times), len(confidence_bins)).
    - times: list, daftar waktu evaluasi dalam detik.
    - pga_thresholds: list, daftar ambang batas PGA dalam %g.
    - save_path: str, path untuk menyimpan gambar (opsional).
    """
    num_rows = len(pga_thresholds)
    num_cols = len(times)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 3 * num_rows), sharex=True, sharey=True)

    for i, pga in enumerate(pga_thresholds):
        for j, t in enumerate(times):
            ax = axes[i, j]

            # Ambil data untuk waktu & threshold ini
            confidence = confidence_bins
            accuracy = accuracy_values[i, j]

            # Plot histogram dengan pola garis diagonal
            ax.bar(confidence, accuracy, width=0.05, color="blue", alpha=0.7, hatch="//", edgecolor="black")

            # Garis referensi (kalibrasi sempurna)
            ax.plot([0, 1], [0, 1], "k--")

            # Label sumbu hanya untuk sisi kiri dan bawah
            if j == 0:
                ax.set_ylabel(f"{pga}%g", fontsize=10)
            if i == num_rows - 1:
                ax.set_xlabel("Confidence", fontsize=10)
            if i == 0:
                ax.set_title(f"{t} s", fontsize=10)

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(True, linestyle="--", alpha=0.5)

    # Tambahkan legenda di luar plot
    fig.legend(["Perfect Calibration", "TEAM"], loc="upper right", fontsize=10)

    plt.tight_layout()

    # Simpan gambar jika path diberikan
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot disimpan di {save_path}")


    
