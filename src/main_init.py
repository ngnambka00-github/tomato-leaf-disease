import hydra
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from omegaconf import OmegaConf

@hydra.main(config_path='../config', config_name='data_path')
def main(config):
    data_path = config.data.leaf_tomato_disease_path
    print(data_path)

if __name__ == "__main__":
    with initialize(config_path="../config"):
        data_cfg = compose(config_name="data_path")

    data_cfg = OmegaConf.create(data_cfg)    

    print("\n")
    print(data_cfg.data.leaf_tomato_disease_path)