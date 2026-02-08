import argparse
import os

from train import train_model
from test import test_model
from utils import *

def main():
    dataset_paths = get_dataset_paths(data_dir, args.dataset)

    if args.generate_density_maps:
        # print("Ordner werden gecleared")
        # clear_folder(dataset_paths["density_train_numpy_dir"])
        # clear_folder(dataset_paths["density_train_tiff_dir"])
        # clear_folder(dataset_paths["image_train_numpy_dir"])
        # clear_folder(dataset_paths["density_test_numpy_dir"])
        # clear_folder(dataset_paths["density_test_tiff_dir"])
        # clear_folder(dataset_paths["image_test_numpy_dir"])
        # print("")

        # print("Anfang Generierung der Train Density Maps")
        # generate_density(dataset_paths["annotations_train_dir"], dataset_paths["image_train_tiff_dir"], dataset_paths["density_train_numpy_dir"], dataset_paths["density_train_tiff_dir"])
        
        # print("")
        # print("Anfang Generierung der Test Density Maps")
        # generate_density(dataset_paths["annotations_test_dir"], dataset_paths["image_test_tiff_dir"], dataset_paths["density_test_numpy_dir"], dataset_paths["density_test_tiff_dir"])

        print("")
        print("Anfang Luminanzbildgenerierung")
        convert_images_to_luminance_train(dataset_paths['dataset_dir'])
        convert_images_to_luminance_test(dataset_paths["dataset_dir"])

        print("")
        print("Anfang Bildskalierung")
        scale_to_fixed_resolution_image(dataset_paths["image_train_numpy_dir"], target_width=1920, target_height=1088)
        scale_to_fixed_resolution_image(dataset_paths["image_test_numpy_dir"], target_width=1920, target_height=1088)

        print("")
        print("Anfang Density Map Skalierung")
        scale_to_fixed_resolution_density(dataset_paths["density_train_numpy_dir"], target_width=1920, target_height=1088)
        scale_to_fixed_resolution_density(dataset_paths["density_test_numpy_dir"], target_width=1920, target_height=1088)


    if args.train:

        model_save_path, checkpoint_path = create_folder_model(model_base_dir, args.resume_training)
        save_args_to_txt(args, os.path.dirname(model_save_path))
        norm_factor = get_biggest_maximum(dataset_paths["density_train_numpy_dir"], model_save_path.replace("image_to_density_map.pth", ""))
        norm_factor = 1

        train_model(dataset_paths['image_train_numpy_dir'], dataset_paths['density_train_numpy_dir'], model_save_path, checkpoint_path, norm_factor, batch_size=args.batch_size, num_epochs=args.num_epochs, patience=args.patience, lr=args.lr, loss_function=args.loss_function, resume_training=args.resume_training)

    if args.test:
        model_path = get_latest_file(model_base_dir)
        test_output_dir = create_folder_test(test_base_dir, model_path)
        save_args_to_txt(args, test_output_dir)
        norm_factor = get_biggest_maximum(dataset_paths["density_train_numpy_dir"], model_path.replace("image_to_density_map.pth", ""))
        norm_factor = 1

        test_model(dataset_paths['image_test_numpy_dir'], model_path, test_output_dir, dataset_paths['density_test_numpy_dir'], norm_factor, batch_size=args.batch_size)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test SDR to density conversion model")
    
    # General arguments
    parser.add_argument('--generate_density_maps',  type=bool,   default=False,  help="Set to True to generate density maps")
    parser.add_argument('--train',                  type=bool,   default=False,   help="Set to True to train the model")
    parser.add_argument('--test',                   type=bool,   default=True,   help="Set to True to test the model")
    
    parser.add_argument('--resume_training',        type=bool,   default=False,  help="Set to True to resume training")

    # Training arguments
    parser.add_argument('--batch_size',             type=int,    default=2,      help="Size of Batch")
    parser.add_argument('--num_epochs',             type=int,    default=1000,   help="Number of training epochs")
    parser.add_argument('--patience',               type=int,    default=5000,   help="Patience for early stopping")
    parser.add_argument('--lr',                     type=float,  default=1e-4,   help="Learning rate for the optimizer")
    parser.add_argument('--loss_function',          type=int,    default=1,      help="1: mse, 2: SSIM, 3: log, 4: mae")
    
    # Dataset selection argument
    parser.add_argument('--dataset',                type=int,    default=1,      help="1 = High_Resolution")
    
    args = parser.parse_args()

    # Base directory for data
    data_dir = r"data"
    
    # Base directories for models and outputs
    model_base_dir = r"model_output"
    test_base_dir = r"image_output"
    
    main()
