import sys
from pathlib import Path
from typing import ClassVar, Any, Literal
from pydantic import BaseModel

from pydantic_settings import (
    BaseSettings,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    YamlConfigSettingsSource,
)


class CliArgsSource(EnvSettingsSource):
    def __init__(
        self, settings_cls: type[BaseSettings], prefix: str | None = None
    ) -> None:
        super().__init__(
            settings_cls, env_prefix=prefix or "", env_nested_delimiter="__"
        )
        self._prefix = prefix or ""
        self.env_vars = self._load_args()

    def _load_args(self) -> dict:
        args = sys.argv[1:]
        env_vars = {}
        for i in range(len(args)):
            if args[i].startswith(f"--{self._prefix}"):
                if "=" in args[i]:
                    key, value = args[i].split("=")
                    key = key[2:].strip()
                    env_vars[key] = value.strip()
                elif i + 1 < len(args) and not args[i + 1].startswith("--"):
                    key = args[i][2:].strip()
                    env_vars[key] = args[i + 1].strip()
        return env_vars


class Settings(BaseSettings):
    _cli_prefix: ClassVar[str] = None
    _yaml_path: ClassVar[str | Path] = None
    _yaml_config_section: ClassVar[str | None] = None

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        sources = [
            init_settings,
            env_settings,
            CliArgsSource(settings_cls, cls._cli_prefix),
        ]
        if cls._yaml_path:
            sources.append(
                YamlConfigSettingsSource(
                    cls,
                    yaml_file=cls._yaml_path,
                    yaml_config_section=cls._yaml_config_section,
                )
            )
        return tuple(sources)


class ConfigSettings(Settings):
    _cli_prefix = "config"
    pointnav_file: str = "pointnav_droid.yaml"


config_settings = ConfigSettings()



class EnvironmentSettings(Settings):
    # _yaml_path = Path.cwd() / "configs" / config_settings.pointnav_file
    # _yaml_config_section = "env"

    frame_width: int = 320
    frame_height: int = 240

    visualize: int = 0  # 1: Render the frame
    print_images: int = 0  # 1: save visualization as images
    vis_type: int = 1  # 1: Show predicted map, 2: Show GT map

    map_size_cm: int = 2400
    map_resolution: int = 5
    du_scale: int = 2
    vision_range: int = 64

    noisy_actions: int = 1
    noisy_odometry: int = 1
    noise_level: float = 1.0

    obs_threshold: float = 1.0
    obstacle_boundary: int = 32
    collision_threshold: float = 0.2

    navmesh_path: str | None = None
    pointcloud_base_slam: Literal["depth", "droid"] = "droid"

    short_goal_dist: int = 2  # max distance between the agent and the short term goal
    skip_invalid_global_goals: bool = True

    def model_post_init(self, context: Any, /) -> None:
        assert self.short_goal_dist >= 1, "args.short_goal_dist >= 1"


env_settings = EnvironmentSettings()

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
    # _yaml_path = Path.cwd() / "configs" / config_settings.pointnav_file
    # _yaml_config_section = "droid_slam"
    _cli_prefix = "droid_slam"

    base_config: DroidConfig = DroidConfig()


droid_slam_settings = DroidSlamSettings()


