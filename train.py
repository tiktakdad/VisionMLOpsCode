import dotenv
import hydra
from omegaconf import DictConfig
from src.training_pipeline import train

dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs/", config_name="train.yaml", version_base=None)
def main(config: DictConfig):
    print(config)
    return train(config)


if __name__ == "__main__":
    main()
