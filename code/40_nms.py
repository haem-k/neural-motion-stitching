import torch
import sys
import argparse
import nms.utils.utils as utils
import nms.utils.network_utils as utils_n
from nms.modules.nms import NeuralMotionStitching

def main(config: argparse.Namespace):
    print("[Neural Motion Stitching]")
    vqvae_train_config = utils.load_config(config.vqvae_train_config_dir, config.vqvae_train_config_name)
    mg_train_config = utils.load_config(config.mg_train_config_dir, config.mg_train_config_name)
    utils_n.set_seeds(vqvae_train_config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    nms = NeuralMotionStitching(config, vqvae_train_config, mg_train_config, device)
    
    print("\nRunning NMS on each sample...")
    result_dir = f"{config.test_result_dir}/{config.exp_name}"
    utils.check_create_dir(result_dir)
    utils.save_config(config, result_dir, "nms")
    test_input = utils.load_config(config.test_input_dir, config.test_input_name)
    nms.generate(
        test_input,
        result_dir,
        combine=config.combine,
        control=config.control,
        blend=config.blend,
        single_frame=config.single_frame,
        save_bvh=config.save_bvh,
    )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="nms")
    args = parser.parse_args()
    config = utils.load_config("./nms/config", args.config)
    
    if config.vqvae_train_config_dir == "" or config.vqvae_train_config_name == "" \
        or config.mg_train_config_dir == "" or config.mg_train_config_name == "":
        print("Please provide config for trained vqvae and motion generator")
        sys.exit()
    
    utils.print_argument(config)
    main(config)
