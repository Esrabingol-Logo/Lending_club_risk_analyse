from src.utils.config import load_config
from src.pipeline.step_00_profile_raw import step_00_profile_raw

if __name__ == "__main__":
    print("Pipeline started...")

    cfg = load_config(
        base_config_path="configs/base_config.yaml",
        dataset_config_path="configs/lending_club.yaml",
    )

    print("Config loaded...")

    out = step_00_profile_raw(cfg)

    print("STEP 00 DONE:", out)
