# import os
# import numpy as np
# import matplotlib.pyplot as plt

# def analyze_annotations(folder_path, x_limit=None):
#     person_counts = []
#     empty_files_count = 0

#     for file_name in os.listdir(folder_path):
#         if file_name.endswith('.npy'):
#             file_path = os.path.join(folder_path, file_name)
#             data = np.load(file_path)
#             person_count = np.count_nonzero(data)
#             person_counts.append(person_count)
#             if person_count == 0:
#                 empty_files_count += 1

#     # Plot the distribution of person counts
#     max_person_count = max(person_counts) if person_counts else 0
#     bins = np.arange(0, max_person_count + 100, 100)  # Bins in 100er Schritten
#     plt.figure(figsize=(16, 8))  # Querformat
#     plt.hist(person_counts, bins=bins, edgecolor='black')
#     plt.xlabel('Personenanzahl', fontsize=18)
#     plt.ylabel('Bilderanzahl', fontsize=18)
#     plt.title('Verteilung der Personenanzahl im Trainingsdatensatz', fontsize=20)
#     if x_limit is not None:
#         plt.xlim(0, x_limit)
#     plt.xticks(fontsize=18)
#     plt.yticks(fontsize=18)
#     plt.show()

#     average_person_count = np.mean(person_counts)
#     std_deviation = np.std(person_counts)
#     print(f"Average number of persons: {average_person_count:.2f}")
#     print(f"Standard deviation of persons: {std_deviation:.2f}")
#     print(f"Number of empty annotation files: {empty_files_count}")

# # Beispielaufruf der Funktion
# annotations_folder_path = r"C:\Users\rippthehorror\Documents\Crowd-Counting-Pipeline\data\High_Resolution\Train_dataset\annotations"
# analyze_annotations(annotations_folder_path, x_limit=22000)

import numpy as np

def sum_numpy_file(file_path):
    """
    Liest eine .npy-Datei ein, summiert die Werte und gibt die Summe aus.

    :param file_path: Pfad zur .npy-Datei
    """
    try:
        data = np.load(file_path)
        total_sum = np.sum(data)
        print(f"Summe der Werte in {file_path}: {total_sum}")
        return total_sum
    except Exception as e:
        print(f"Fehler beim Einlesen der Datei {file_path}: {e}")
        return None

# Beispielaufruf der Funktion
file_path = r"C:\Users\rippthehorror\Desktop\data\High_Resolution\Train_dataset\density_train_numpy\jhu_crowd_fhd_1_density.npy"
sum_numpy_file(file_path)