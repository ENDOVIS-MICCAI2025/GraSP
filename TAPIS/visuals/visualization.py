import os
import re
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ---------------------------------------------------------------------------- #
# For processing the JSON files (annotations and predictions)
# ---------------------------------------------------------------------------- #

def load_json_file(file):
    """This function loads a json file and returns its content."""
    with open(file, 'r') as json_file:
        return json.load(json_file)

def sort_keys_numerically(keys):
    """Sort dictionary keys numerically based on the video number."""
    def extract_number(key):
        numbers = re.findall(r'\d+', key)
        return int(numbers[-1]) if numbers else 0
    
    return sorted(keys, key=extract_number)

def process_annotations(ann_dict):
    result = {}

    for ann in ann_dict:
        # Extraemos el nombre del video (video_XXX) y el frame (frameXXX)
        video_name = ann['image_name'].split("/")[7]  # video_XXX
        frame_name = ann['image_name'].split("/")[8]  # 00001, 00002, etc.

        # Creamos la clave 'video_XXX/frameXXX'
        key = f"{video_name}/{frame_name}"

        # Asignamos el valor de phases
        result[key] = ann['phases']

    return result

def standardize_video_name(video_name):
    """
    Estandariza el nombre del video eliminando ceros iniciales si tiene 2 dígitos
    y manteniéndolo cuando tiene 3 dígitos.
    """
    # Extraer el número del video, ignorando 'video_' y ceros al principio
    video_number = int(video_name.replace('video_', ''))  # Convierte a número para quitar ceros al inicio
    return f"video_{video_number}"  # Retorna el nombre con el formato correcto

# ---------------------------------------------------------------------------- #
# For converting annotations and predictions to RGB and plotting them
# ---------------------------------------------------------------------------- #

def hex_to_rgb(hex_color):
    """Convert a color in hex format to RGB."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def rgb_annotations(color_tuple, annotations):
    """Convert annotation values to RGB."""
    annotations_array = np.array(annotations)
    return np.array([color_tuple[ann] for ann in annotations_array])

def plot_combined_anns_preds(annotations, predictions_dict, phases_colors, output_filename, video):
    """Plot ground truth and multiple predictions side by side."""
    # Convert ground truth annotations to RGB
    annotations_rgb = rgb_annotations([hex_to_rgb(color) for color in phases_colors], annotations)

    # Prepare matrices for ground truth and predictions
    matrix_annotations = np.ones((int(len(annotations_rgb) * 0.2), len(annotations_rgb), 3))
    matrix_annotations *= annotations_rgb.reshape(1, -1, 3)

    # Create and save ground truth figure
    fig_ann, ax_ann = plt.subplots(figsize=(10, 2))
    ax_ann.imshow(matrix_annotations)
    ax_ann.axis('off')
    ax_ann.set_title(f"Ground Truth (GT) - {video}")
    os.makedirs(video, exist_ok=True)
    plt.savefig(f"{video}/{video}_ground_truth_{output_filename}", dpi=800, bbox_inches='tight')
    plt.close(fig_ann)

    # Process each prediction type
    for pred_name, predictions in predictions_dict.items():
        predictions_rgb = rgb_annotations([hex_to_rgb(color) for color in phases_colors], predictions)
        matrix_predictions = np.ones((int(len(predictions_rgb) * 0.2), len(predictions_rgb), 3))
        matrix_predictions *= predictions_rgb.reshape(1, -1, 3)

        # Create and save prediction figure
        fig_pred, ax_pred = plt.subplots(figsize=(10, 2))
        ax_pred.imshow(matrix_predictions)
        ax_pred.axis('off')
        ax_pred.set_title(f"Predictions ({pred_name}) - {video}")
        plt.savefig(f"{video}/{video}_predictions_{pred_name}_{output_filename}", dpi=800, bbox_inches='tight')
        plt.close(fig_pred)

def plot_video_anns_preds(anns_dict, preds_dicts, phases_colors, video=None):
    """Plot annotations and predictions for a video."""
    plot_combined_anns_preds(anns_dict, preds_dicts, phases_colors, f"{video}_visual.png", video)

# def plot_combined_anns_preds(annotations, predictions, phases_colors, output_filename, video):
#     """Plot both annotations and predictions on separate images."""
#     # Convert both annotations and predictions to RGB values
#     annotations_rgb = rgb_annotations([hex_to_rgb(color) for color in phases_colors], annotations)
#     predictions_rgb = rgb_annotations([hex_to_rgb(color) for color in phases_colors], predictions)

#     # Create matrices for both annotations and predictions
#     matrix_predictions = np.ones((int(len(predictions_rgb) * 0.2), len(predictions_rgb), 3))
#     matrix_annotations = np.ones((int(len(annotations_rgb) * 0.2), len(annotations_rgb), 3))

#     # Fill the matrices with corresponding RGB values
#     matrix_predictions *= predictions_rgb.reshape(1, -1, 3)
#     matrix_annotations *= annotations_rgb.reshape(1, -1, 3)

#     # Create a figure to plot predictions
#     fig_pred, ax_pred = plt.subplots(figsize=(10, 2))
#     ax_pred.imshow(matrix_predictions)
#     ax_pred.axis('off')
#     ax_pred.set_title(f"Predictions - {video}")
    
#     # Save the prediction image
#     os.makedirs(video, exist_ok=True)
#     plt.savefig(f"{video}/{video}_predictions_{output_filename}", dpi=800, bbox_inches='tight')
#     plt.close(fig_pred)

#     # Create a figure to plot annotations (GT)
#     fig_ann, ax_ann = plt.subplots(figsize=(10, 2))
#     ax_ann.imshow(matrix_annotations)
#     ax_ann.axis('off')
#     ax_ann.set_title(f"Ground Truth (GT) - {video}")
    
#     # Save the annotation image
#     plt.savefig(f"{video}/{video}_annotations_{output_filename}", dpi=800, bbox_inches='tight')
#     plt.close(fig_ann)

# def plot_video_anns_preds(anns_dict, preds_dict, phases_colors, video=None):
#     """Plot annotations and predictions side by side for each video."""
    
#     # Call the function to plot combined annotations and predictions
#     plot_combined_anns_preds(anns_dict, preds_dict, phases_colors, f"{video}_visual.png", video)

def main():

    video_names = [
    "video_015", "video_016", "video_017", "video_018", "video_019", "video_020", 
    "video_021", "video_062", "video_063", "video_064", "video_065", "video_066", 
    "video_067", "video_068", "video_069", "video_070", "video_071", "video_072", 
    "video_073", "video_074", "video_075", "video_076", "video_077", "video_078", 
    "video_079", "video_080", "video_081", "video_082", "video_083", "video_084", 
    "video_085", "video_086", "video_087", "video_088", "video_089", "video_090", 
    "video_091", "video_092", "video_093", "video_094", "video_095", "video_096", 
    "video_097", "video_098", "video_099", "video_100", "video_101", "video_118", 
    "video_119", "video_120", "video_121", "video_122", "video_123", "video_124", 
    "video_125", "video_133", "video_134", "video_135", "video_143", "video_144", 
    "video_145", "video_153", "video_154", "video_155", "video_183", "video_184", 
    "video_185", "video_186", "video_187", "video_188", "video_189", "video_190", 
    "video_191", "video_192", "video_193", "video_194", "video_195", "video_196"
    ]

    # Define the colors in hex format
    phases_colors = [
        "#434370", "#0081d0", "#83b6ff", "#8388f4", "#98c4ff", "#004aad", "#caf5ff", "#ba2d53",
        "#bf6d83", "#e14a73", "#5e3967", "#ffb7ca", "#820023", "#bda7d9", "#ffd493", "#00c178",
        "#395ca9", "#f49c6d", "#e9c630", "#d1ef34", "#e69c3c", "#fbaaa0", "#59e942", "#00cfa9",
        "#15ae88", "#1d5562", "#24728c", "#0c6092"
        ]

    ann = "/home/naparicioc/ENDOVIS/GraSP/TAPIS/data/Levis/annotations/test.json"
    preds_mvitv1 = "/home/naparicioc/ENDOVIS/GraSP/TAPIS/outputs/Levis/PHASES/ALL_TAPIS_Padding/best_predictions/best_all_13_preds_phases.json"
    preds_mvitv2 = "/home/naparicioc/ENDOVIS/GraSP/TAPIS/outputs/Levis/PHASES/ALL_TAPIS_v2_Padding/best_predictions/best_13_preds_phases.json"
    preds_logits = '/home/naparicioc/ENDOVIS/GraSP/TAPIS/outputs/Levis/PHASES/ALL_TAPIS_v2_Padding_Logits/best_predictions/best_all_9_preds_phases.json'

    ann_dict = load_json_file(ann)['annotations']
    preds_mvitv1_dict = load_json_file(preds_mvitv1)
    preds_mvitv2_dict = load_json_file(preds_mvitv2)
    preds_logits_dict = load_json_file(preds_logits)

    sorted_anns_dict = process_annotations(ann_dict)
    sorted_preds_mvitv1 = dict(sorted(preds_mvitv1_dict.items()))
    sorted_preds_mvitv2 = dict(sorted(preds_mvitv2_dict.items()))
    sorted_preds_logits = dict(sorted(preds_logits_dict.items()))

    for video in tqdm(video_names):
        gt = []
        preds = {
            "mvit": [],
            "mvitv2": [],
            "logits": []
        }

        for key in sorted_preds_mvitv1.keys():
            if video in key:
                video_num, frame_num = key.split("/")
                video_ann = standardize_video_name(video_num)
                index_ann = f"{video_ann}/{frame_num}"

                try:
                    gt.append(sorted_anns_dict[index_ann])
                    preds["mvit"].append(np.argmax(sorted_preds_mvitv1[key]["phases_score_dist"]))
                    preds["mvitv2"].append(np.argmax(sorted_preds_mvitv2[key]["phases_score_dist"]))
                    preds["logits"].append(np.argmax(sorted_preds_logits[key]["phases_score_dist"]))
                except KeyError:
                    index_ann = index_ann.replace("jpg", "png")
                    gt.append(sorted_anns_dict[index_ann])
                    preds["mvit"].append(np.argmax(sorted_preds_mvitv1[key]["phases_score_dist"]))
                    preds["mvitv2"].append(np.argmax(sorted_preds_mvitv2[key]["phases_score_dist"]))
                    preds["logits"].append(np.argmax(sorted_preds_logits[key]["phases_score_dist"]))

        plot_video_anns_preds(gt, preds, phases_colors, video=video)

    # ann_dict = load_json_file(ann)['annotations']
    # preds_dict = load_json_file(preds)

    # sorted_preds_dict = dict(sorted(preds_dict.items()))
    # sorted_anns_dict = process_annotations(ann_dict)

    # for video in tqdm(video_names):
    #     gt = []
    #     pred = []

    #     for key in sorted_preds_dict.keys():
    #         if video in key:
    #             video_num, frame_num = key.split("/")

    #             video_ann = standardize_video_name(video_num)

    #             index_ann = f"{video_ann}/{frame_num}"

    #             # Obtener la anotación correspondiente
    #             try:
    #                 gt.append(sorted_anns_dict[index_ann])  # Asegúrate de que existe la clave
    #                 pred.append(np.argmax(sorted_preds_dict[key]["phases_score_dist"]))  # Agregar la predicción correspondiente
                
    #             except:
    #                 index_ann = index_ann.replace("jpg", "png")
    #                 gt.append(sorted_anns_dict[index_ann])
    #                 pred.append(np.argmax(sorted_preds_dict[key]["phases_score_dist"]))

    #     plot_video_anns_preds(gt, pred, phases_colors, video=video)

if __name__ == "__main__":
    main()
