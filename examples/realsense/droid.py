from pathlib import Path

from pydantic import BaseModel

from base_settings import Settings, config_settings


class DroidConfig(BaseModel):
    # image_dir: str = "./tmp/droid/imgs"
    t0: int = 0
    stride: int = 1

    weights: str = "./pretrained_models/droid.pth"
    buffer: int = 1024
    disable_vis: bool = False

    beta: float = 0.3
    filter_thresh: float = 0.2
    warmup: int = 4
    keyframe_thresh: float = 3.0
    frontend_thresh: float = 16.0
    frontend_window: int = 25
    frontend_radius: int = 2
    frontend_nms: int = 1
    frontend_device: str = "cuda"

    backend_thresh: float = 22.0
    backend_radius: int = 2
    backend_nms: int = 3
    asynchrnous: bool = False
    backend_device: str = "cuda"

    reconstruction_path: str | None = None
    upsample: bool = False
    stereo: bool = False


class DroidSlamSettings(Settings):
    _yaml_path = Path.cwd() / "configs" / config_settings.pointnav_file
    _yaml_config_section = "droid_slam"
    _cli_prefix = "droid_slam"

    base_config: DroidConfig = DroidConfig()


droid_slam_settings = DroidSlamSettings()
