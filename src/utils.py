import os
from datetime import datetime
import glob
import matplotlib.pyplot as plt
import shutil
import tifffile as tiff
import numpy as np
from PIL import Image
from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter
import scipy.io as sio
import tkinter as tk
from tkinter import filedialog

def create_folder_test(base_dir, model_path):

    model_dir = os.path.dirname(model_path)  # Hol den Ordnerpfad
    model_folder_name = os.path.basename(model_dir)  # Hol den Ordnernamen
    timestamp = model_folder_name.split('model_')[1]  # Nimm den Zeitstempel aus dem Ordnernamen

    output_dir = os.path.join(base_dir, f"predicted_density_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    return output_dir

def create_folder_model(base_dir, resume_training=False):
    os.makedirs(base_dir, exist_ok=True)
    current_date = datetime.now().strftime('%Y%m%d_%H%M')
    model_save_dir = os.path.join(base_dir, f"model_{current_date}")
    
    checkpoint_path = 0
    if resume_training == True:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        checkpoint_path = filedialog.askopenfilename(
            initialdir=base_dir,
            title="Select Checkpoint File",
            filetypes=(("PyTorch Checkpoints", "*.pt *.pth"), ("All Files", "*.*"))
        )
        checkpoint_dir = os.path.dirname(checkpoint_path)
        model_save_dir_resume = checkpoint_dir + "_continued"
        model_save_path = os.path.join(model_save_dir_resume, 'image_to_density_map.pth')
        os.makedirs(model_save_dir_resume, exist_ok=True)
    else:
        os.makedirs(model_save_dir, exist_ok=True)
        model_save_path = os.path.join(model_save_dir, 'image_to_density_map.pth')

    return model_save_path, checkpoint_path

def get_latest_file(dir):
   
    # Suche nach allen Dateien im Verzeichnis, die dem Muster entsprechen
    files = glob.glob(os.path.join(dir, "model_*", "image_to_density_map.pth"))

    # Überprüfen, ob Dateien gefunden wurden
    if not files:
        raise FileNotFoundError("No model files found.")

    # Sortiere die Dateien nach dem Datum und der Uhrzeit, die im Dateipfad enthalten sind
    files.sort(key=os.path.getmtime, reverse=True)

    # Die aktuellste Datei ist nun die erste in der Liste
    latest_path = files[0]

    # Beispiel für die Ausgabe des Pfads der aktuellsten Datei
    print(f"The latest model path is: {latest_path}")

    return latest_path

def get_latest_folder(dir, model):
   
    # Suche nach allen Dateien im Verzeichnis, die dem Muster entsprechen
    if model == 1:
        files = glob.glob(os.path.join(dir, "predicted_density_luminance_*"))
    elif model ==2:
        files = glob.glob(os.path.join(dir, "predicted_density_colour_*"))

    # Überprüfen, ob Dateien gefunden wurden
    if not files:
        raise FileNotFoundError("No model folder found.")

    # Sortiere die Dateien nach dem Datum und der Uhrzeit, die im Dateipfad enthalten sind
    files.sort(key=os.path.getmtime, reverse=True)

    # Die aktuellste Datei ist nun die erste in der Liste
    latest_path = files[0]

    # Beispiel für die Ausgabe des Pfads der aktuellsten Datei
    print(f"The latest test path is: {latest_path}")

    return latest_path



def save_args_to_txt(args, output_dir):
    file_path = os.path.join(output_dir, "arguments.txt")  # Ergänze den Dateinamen zum Pfad
    with open(file_path, 'w') as f:
        for arg, value in vars(args).items():
            f.write(f'{arg}: {value}\n')
    print(f"Arguments saved to {file_path}")


# Funktion, um den Ordner zu leeren
def clear_folder(folder_path):
    if os.path.exists(folder_path):
        # Löscht alle Dateien und Unterordner im Ordner
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Datei oder Symbolischer Link löschen
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Ordner und Inhalt löschen
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        os.makedirs(folder_path)  # Ordner erstellen, falls er nicht existiert


def copy_folder_contents(source_folder, destination_folder):
    # Sicherstellen, dass das Zielverzeichnis existiert
    os.makedirs(destination_folder, exist_ok=True)

    # Iteriere durch alle Dateien und Unterordner im Quellordner
    for item in os.listdir(source_folder):
        source_path = os.path.join(source_folder, item)
        destination_path = os.path.join(destination_folder, item)

        # Wenn es sich um eine Datei handelt, kopiere sie
        if os.path.isfile(source_path):
            shutil.copy2(source_path, destination_path)
        # Wenn es sich um ein Verzeichnis handelt, kopiere den gesamten Ordner
        elif os.path.isdir(source_path):
            shutil.copytree(source_path, destination_path, dirs_exist_ok=True)



def Img_to_Luminance(image_file):
    # Bild laden
    image = Image.open(image_file)
    image_arr = np.array(image, dtype=np.float32) / 255.0  # Normalisierung auf [0, 1]
    
    # Überprüfen, ob das Bild bereits ein Luminanzbild ist
    if image_arr.ndim == 2 or (image_arr.ndim == 3 and image_arr.shape[2] == 1):
        # Bild ist bereits ein Luminanzbild
        luminance = image_arr
    else:
        # Luminanz aus RGB-Bild extrahieren (0.299 * R + 0.587 * G + 0.114 * B)
        luminance = 0.299 * image_arr[..., 0] + 0.587 * image_arr[..., 1] + 0.114 * image_arr[..., 2]
    
    # Luminanzbereich ausgeben (optional)
    # print(f"Luminance - Min: {luminance.min()}, Max: {luminance.max()}")
    
    return luminance


def convert_images_to_luminance_train(data_dir):
    train_dir = os.path.join(data_dir, "Train_dataset")

    image_train_colour_dir = os.path.join(train_dir, "image_train_tiff")
    density_train_colour_dir = os.path.join(train_dir, "density_train_tiff")

    image_train_luminance_dir = os.path.join(train_dir, "image_train_numpy")
    density_train_luminance_dir = os.path.join(train_dir, "density_train_numpy")

    
    # Ordner machen
    os.makedirs(image_train_luminance_dir, exist_ok=True)
    os.makedirs(density_train_luminance_dir, exist_ok=True)

    # Ordner leeren
    clear_folder(image_train_luminance_dir)
    # clear_folder(density_train_luminance_dir)

    # image-Train-TIFFs in Luminanz umwandeln und speichern
    image_train_paths = glob.glob(os.path.join(image_train_colour_dir, '*.jpg'))

    print(image_train_paths)
    for image_train_path in image_train_paths:
        image_train_bw = Img_to_Luminance(image_train_path)
        image_train_luminance_path = os.path.join(image_train_luminance_dir, os.path.basename(image_train_path).replace('.jpg', '_luminance.npy'))
        np.save(image_train_luminance_path, image_train_bw)
        print(f"Luminanzbilderstellung für {os.path.basename(image_train_luminance_path)} abgeschlossen")
    
    # density-Train-TIFFs in Luminanz umwandeln und speichern
    # density_train_paths = glob.glob(os.path.join(density_train_colour_dir, '*.tiff'))
    # for density_train_path in density_train_paths:
    #     # density_train_disp_light = Img_to_Luminance(density_train_path)
    #     image = Image.open(density_train_path)
    #     density_train_disp_light = np.array(image, dtype=np.float32) / 255.0
    #     density_train_luminance_path = os.path.join(density_train_luminance_dir, os.path.basename(density_train_path).replace('.tiff', '_luminance.npy'))
    #     np.save(density_train_luminance_path, density_train_disp_light)
    #     print(f"density Train Luminance for {os.path.basename(density_train_luminance_path)} saved")

def convert_images_to_luminance_test(data_dir):
    test_dir = os.path.join(data_dir, "Test_dataset")

    image_test_colour_dir = os.path.join(test_dir, "image_test_tiff")
    density_test_colour_dir = os.path.join(test_dir, "density_test_tiff")

    image_test_luminance_dir = os.path.join(test_dir, "image_test_numpy")
    density_test_luminance_dir = os.path.join(test_dir, "density_test_numpy")

    # Ordner machen
    os.makedirs(image_test_luminance_dir, exist_ok=True)
    os.makedirs(density_test_luminance_dir, exist_ok=True)

    # Ordner leeren
    clear_folder(image_test_luminance_dir)
    # clear_folder(density_test_luminance_dir)

    # image-Test-TIFFs in Luminanz umwandeln und speichern
    image_test_paths = glob.glob(os.path.join(image_test_colour_dir, '*.jpg'))
    for image_test_path in image_test_paths:
        image_test_disp_light = Img_to_Luminance(image_test_path)
        image_test_luminance_path = os.path.join(image_test_luminance_dir, os.path.basename(image_test_path).replace('.jpg', '_luminance.npy'))
        np.save(image_test_luminance_path, image_test_disp_light)
        print(f"Luminanzbilderstellung für {os.path.basename(image_test_luminance_path)} abgeschlossen")

    # # density-Test-TIFFs in Luminanz umwandeln und speichern
    # density_test_paths = glob.glob(os.path.join(density_test_colour_dir, '*.tiff'))
    # for density_test_path in density_test_paths:
    #     image = Image.open(density_test_path)
    #     density_test_disp_light = np.array(image, dtype=np.float32) / 255.0
    #     density_test_luminance_path = os.path.join(density_test_luminance_dir, os.path.basename(density_test_path).replace('.tiff', '_luminance.npy'))
    #     np.save(density_test_luminance_path, density_test_disp_light)
    #     print(f"Density Test Luminance for {os.path.basename(density_test_luminance_path)} saved")

# def scale_to_fixed_resolution(input_dir, target_width, target_height):
#     """
#     Skaliert alle Bilder in einem Ordner auf eine feste Auflösung (1920x1080),
#     wobei die Skalierung die Breite oder Höhe anpasst und der Rest mit Padding aufgefüllt wird.
#     """
#     for filename in os.listdir(input_dir):
#         if filename.endswith('.npy'):
#             file_path = os.path.join(input_dir, filename)
#             print(f"Processing {file_path}")

#             # Lade das Bild als numpy-Array
#             array = np.load(file_path)

#             original_sum = np.sum(array)

#             # print(array.shape)

#             # Berechne die aktuelle Höhe und Breite des Arrays
#             current_height, current_width = array.shape

#             # Bestimme den Skalierungsfaktor für Höhe oder Breite
#             if current_width / target_width > current_height / target_height:
#                 # Skalieren der Breite auf target_width
#                 scale_factor = target_width / current_width
#             else:
#                 # Skalieren der Höhe auf target_height
#                 scale_factor = target_height / current_height

#             # Skaliere das Bild
#             scaled_array = zoom(array, (scale_factor, scale_factor), order=3)
#             scale_factor = (current_width * current_height) / (target_width * target_height)
#             scaled_array = scaled_array * scale_factor

#             # Padding berechnen, falls notwendig
#             padded_array = pad_image(scaled_array, target_width, target_height)
#             # plt.imshow(padded_array, cmap='gray')  # Wenn das Bild ein Graustufenbild ist, sonst entferne `cmap='gray'`
#             # plt.title(f"")
#             # plt.axis('off')  # Achsen ausblenden
#             # plt.show()
#             # Speichere das Ergebnis
#             np.save(file_path, padded_array)

#             print(f"Originale Summe: {original_sum}, Summe nach rescaling: {np.sum(padded_array)}")
#             print(f"Saved scaled and padded image to {file_path}")
            
# def pad_image(image, target_width=1920, target_height=1080):
#     """
#     Fügt Padding hinzu, um das Bild auf die Zielauflösung (Full HD) zu bringen.
#     """
#     current_height, current_width = image.shape

#     # Berechne die Anzahl der Paddings in x- und y-Richtung
#     pad_height = target_height - current_height
#     pad_width = target_width - current_width

#     # Padding zu den Seiten hinzufügen
#     padding_top = pad_height // 2
#     padding_bottom = pad_height - padding_top
#     padding_left = pad_width // 2
#     padding_right = pad_width - padding_left

#     # Padding anwenden
#     padded_image = np.pad(image, ((padding_top, padding_bottom), (padding_left, padding_right)), mode='constant', constant_values=0)

#     return padded_image

def scale_to_fixed_resolution_image(input_dir, target_width, target_height):
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.npy'):
            file_path = os.path.join(input_dir, file_name)
            image = np.load(file_path)

            # Resize the density map
            image_scaled = np.array(Image.fromarray(image).resize((target_width, target_height), Image.BILINEAR))

            # Save the resized density map
            np.save(file_path, image_scaled)
        print(f"Scaling für {file_name} abgeschlossen")

def scale_to_fixed_resolution_density(input_dir, target_width, target_height):
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.npy'):
            file_path = os.path.join(input_dir, file_name)
            density_map = np.load(file_path)
            original_sum = np.sum(density_map)

            # Resize the density map
            density_map_resized = np.array(Image.fromarray(density_map).resize((target_width, target_height), Image.BILINEAR))

            # print(f"Original sum: {original_sum}, Resized sum: {np.sum(density_map_resized)}")
            # Adjust the sum of the resized density map to match the original sum
            resized_sum = np.sum(density_map_resized)
            if resized_sum != 0:
                density_map_resized = density_map_resized * (original_sum / resized_sum)

            print(f"Originale Summe: {original_sum}, Summe nach rescaling: {np.sum(density_map_resized)}")

            # Save the resized density map
            np.save(file_path, density_map_resized)
            print(f"Scaling für {file_name} abgeschlossen")




def get_dataset_paths(data_dir, dataset_option):
    """Returns the appropriate paths for the selected dataset."""
    dataset_names = {
        1: 'High_Resolution',
    }

    if dataset_option not in dataset_names:
        raise ValueError(f"Dataset option {dataset_option} not recognized. Choose between 1 and {len(dataset_names)}.")
    
    dataset_name = dataset_names[dataset_option]
    dataset_dir = os.path.join(data_dir, dataset_name)

    # Train dataset directories
    train_dir = os.path.join(dataset_dir, 'Train_dataset')
    image_train_tiff_dir = os.path.join(train_dir, 'image_train_tiff')
    image_train_numpy_dir = os.path.join(train_dir, 'image_train_numpy')
    density_train_tiff_dir = os.path.join(train_dir, 'density_train_tiff')
    density_train_numpy_dir = os.path.join(train_dir, 'density_train_numpy')
    annotations_train_dir = os.path.join(train_dir, 'annotations')
    
    # Test dataset directories
    test_dir = os.path.join(dataset_dir, 'Test_dataset')
    image_test_numpy_dir = os.path.join(test_dir, 'image_test_numpy')
    image_test_tiff_dir = os.path.join(test_dir, 'image_test_tiff')
    density_test_numpy_dir = os.path.join(test_dir, 'density_test_numpy')
    density_test_tiff_dir = os.path.join(test_dir, 'density_test_tiff')
    annotations_test_dir = os.path.join(test_dir, 'annotations')

    return {
        "dataset_dir": dataset_dir,
        "image_train_tiff_dir": image_train_tiff_dir,
        "image_train_numpy_dir": image_train_numpy_dir,
        "density_train_tiff_dir": density_train_tiff_dir,
        "density_train_numpy_dir": density_train_numpy_dir,
        "image_test_numpy_dir": image_test_numpy_dir,
        "density_test_numpy_dir": density_test_numpy_dir,
        "image_test_tiff_dir": image_test_tiff_dir,
        "density_test_tiff_dir": density_test_tiff_dir,
        "annotations_train_dir": annotations_train_dir,
        "annotations_test_dir": annotations_test_dir
    }


def save_density_map_as_tiff(density_map, output_path):
    density_map_8bit = (255 * (density_map / np.max(density_map))).astype(np.uint8)
    img = Image.fromarray(density_map_8bit)
    img.save(output_path, format='TIFF')

def get_biggest_maximum(density_dir, txt_dir):
    # Initialisieren einer Liste für Maxima
    all_maxima = []

    txt_path = os.path.join(txt_dir, "biggest_maximum.txt")

    # Alle Dateien im Ordner durchgehen
    for numpy_name in os.listdir(density_dir):
        if numpy_name.endswith(".npy"):  # Nur Dateien mit der Endung .npy berücksichtigen
            datei_pfad = os.path.join(density_dir, numpy_name)
            try:
                # NumPy-Array laden
                array = np.load(datei_pfad)
                # Maximum des aktuellen Arrays bestimmen
                max_wert = np.max(array)
                all_maxima.append(max_wert)
                # print(f"Maximum in {numpy_name}: {max_wert}")
            except Exception as e:
                print(f"Fehler beim Lesen von {numpy_name}: {e}")

    # Das größte Maximum bestimmen
    if all_maxima:
        biggest_maximum = np.max(all_maxima)
        print(f"\nThe biggest maximum is: {biggest_maximum}")

        # Ergebnis in eine Textdatei schreiben
        with open(txt_path, "w") as file:
            file.write(f"{biggest_maximum}\n")

        # print(f"Ergebnis wurde in '{txt_path}' gespeichert.")
    else:
        print("Keine gültigen NumPy-Dateien gefunden oder die Dateien waren leer.")

    return biggest_maximum




def generate_density(annotations_dir, image_dir, density_numpy_dir, density_tiff_dir):
    def create_adaptive_density_map(annotations, image_shape, avg_distances, beta=0.3):
        density_map = np.zeros(image_shape, dtype=np.float32)

        for idx, (point, avg_dist) in enumerate(zip(annotations, avg_distances)):
            x, y = int(point[0]), int(point[1])  # Runden der Koordinaten
            if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
                if np.isnan(avg_dist):
                    sigma = 100  # Fester Wert für den Fall, dass avg_dist NaN ist
                else:
                    sigma = beta * avg_dist

                size = max(1, int(6 * sigma))  # Minimale Größe sicherstellen
                x_min, x_max = max(0, x - size), min(image_shape[1], x + size)
                y_min, y_max = max(0, y - size), min(image_shape[0], y + size)

                local_density = np.zeros((y_max - y_min, x_max - x_min), dtype=np.float32)
                local_density[y - y_min, x - x_min] = 1
                local_density = gaussian_filter(local_density, sigma=sigma, mode='constant')

                density_map[y_min:y_max, x_min:x_max] += local_density
        return density_map

    def calculate_distances(annotations, k_neighbors=3):
        num_points = len(annotations)
        avg_distances = np.zeros(num_points, dtype=np.float32)

        if num_points == 1:
            avg_distances[0] = np.nan  # Markiere als NaN, wenn es nur eine Annotation gibt
        else:
            for i, point in enumerate(annotations):
                distances = np.sqrt(np.sum((annotations - point) ** 2, axis=1))
                sorted_distances = np.sort(distances)[1:]  # Entferne die Distanz zu sich selbst
                k = min(k_neighbors, len(sorted_distances))
                avg_distances[i] = np.mean(sorted_distances[:k])

        return avg_distances

    beta = 0.3
    k_neighbors = 3

    deviations = []

    # Verarbeite alle Annotationsdateien im Ordner
    for annotation_file in sorted(os.listdir(annotations_dir)):
        if annotation_file.endswith('.npy'):
            annotation_path = os.path.join(annotations_dir, annotation_file)
            base_name = os.path.splitext(annotation_file)[0]

            # Lade Annotationen und das zugehörige Bild
            annotations = np.load(annotation_path)

            # Finde das zugehörige Bild
            image_path = os.path.join(image_dir, base_name.replace("_ann", "") + ".jpg")
            if not os.path.exists(image_path):
                print(f"Bild für {annotation_file} nicht gefunden: {image_path}")
                continue

            img = Image.open(image_path)
            image_shape = (img.size[1], img.size[0])  # Höhe, Breite

            print(f"Verarbeite {annotation_file} und {image_path}...")

            # Berechne die adaptive Density Map
            avg_distances = calculate_distances(annotations, k_neighbors)
            adaptive_density_map = create_adaptive_density_map(annotations, image_shape, avg_distances, beta)

            # Speichere die Ergebnisse
            npy_output_path = os.path.join(density_numpy_dir, f"{base_name}_density.npy")
            tiff_output_path = os.path.join(density_tiff_dir, f"{base_name}_density.tiff")

            np.save(npy_output_path, adaptive_density_map)
            save_density_map_as_tiff(adaptive_density_map, tiff_output_path)

            total_annotations = len(annotations)
            adaptive_total = np.sum(adaptive_density_map)

            deviation = np.abs(adaptive_total - total_annotations) / total_annotations * 100
            deviations.append(deviation)

#            print(f"Dichtekarte gespeichert: {npy_output_path} und {tiff_output_path}")

            # Ausgabe der Gesamtanzahl der Annotationen und Summe der Density Map
            total_annotations = len(annotations)
            adaptive_total = np.sum(adaptive_density_map)
            print(f"Anzahl Annotationen: {total_annotations}, Integrierte Summe: {adaptive_total:.2f}")

    mean_deviation = np.mean(deviations)
    std_deviation = np.std(deviations)

    parent_dir = os.path.dirname(annotations_dir)
    output_stats_path = os.path.join(parent_dir, "density_stats.txt")

    with open(output_stats_path, 'w') as f:
        for deviation in deviations:
            f.write(f"{deviation:.2f}%\n")
        f.write(f"\nMittelwert der Abweichungen: {mean_deviation:.2f}%\n")
        f.write(f"Standardabweichung der Abweichungen: {std_deviation:.2f}%\n")
    print("------------------------------------")
    print(f"Mittelwert der Abweichungen: {mean_deviation:.2f}%")
    print(f"Standardabweichung der Abweichungen: {std_deviation:.2f}%")
    print("Density Map Generation abgeschlossen!")



# def compare_annotations_with_density_folders(annotations_dir, density_maps_dir):
#     deviations = []
#     total_files = len([name for name in os.listdir(annotations_dir) if name.endswith('.npy')])
#     processed_files = 0

#     print(f"Starting comparison of annotations and density maps in '{annotations_dir}' and '{density_maps_dir}'")
# #    print(f"Total files to process: {total_files}")

#     for file_name in os.listdir(annotations_dir):
#         if file_name.endswith('.npy'):
#             annotations_path = os.path.join(annotations_dir, file_name)
#             density_map_file_name = file_name.replace('.npy', '_density.npy')  # Anpassung des Dateinamens
#             density_map_path = os.path.join(density_maps_dir, density_map_file_name)

#             if os.path.exists(density_map_path):
#                 # print(f"Processing file: {file_name}")
#                 annotations = np.load(annotations_path)
#                 density_map = np.load(density_map_path)

#                 num_annotations = np.count_nonzero(annotations)
#                 density_sum = np.sum(density_map)

#                 print(f"File: {file_name}, Annotations: {num_annotations}, Density Sum: {density_sum}")

#                 if num_annotations > 0:
#                     deviation = abs(num_annotations - density_sum) / num_annotations * 100
#                     deviations.append(deviation)
#             else:
#                 print(f"Density map not found for file: {file_name}")

#             processed_files += 1
#             # print(f"Processed {processed_files}/{total_files} files", end='\r')

#     # print("\nFinished processing files.")

#     if deviations:
#         mean_deviation = np.mean(deviations)
#         std_deviation = np.std(deviations)
#     else:
#         mean_deviation = float('nan')
#         std_deviation = float('nan')

#     parent_dir = os.path.dirname(annotations_dir)
#     output_stats_path = os.path.join(parent_dir, "density_stats_after_resize.txt")

#     with open(output_stats_path, 'w') as f:
#         for deviation in deviations:
#             f.write(f"{deviation:.2f}%\n")
#         f.write(f"\nMittelwert der Abweichungen: {mean_deviation:.2f}%\n")
#         f.write(f"Standardabweichung der Abweichungen: {std_deviation:.2f}%\n")

#     print(f"Mittelwert der Abweichungen: {mean_deviation:.2f}%")
#     print(f"Standardabweichung der Abweichungen: {std_deviation:.2f}%")
#     print(f"Results saved to '{output_stats_path}'")
#     print("Comparison of Annotations and Density Maps completed!")