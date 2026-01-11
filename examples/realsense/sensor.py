from typing import Any
from enum import Enum


class Sensor(Enum):
    RGB = "rgb"
    DEPTH = "depth"  # cm
    POSE = "pose"  # position & rotation
    RGB_RIGHT = "rgb_right"
    STEREO = "stereo"


SensorData = dict[Sensor, Any]
