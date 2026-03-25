from PIL import Image
import pyautogui
import time
import configparser

import cv2
import numpy as np
from pynput import keyboard as pynput_keyboard

CONFIG_FILE = "config.ini"
DEFAULT_DELAYS = {
    "key_press_delay": 0.1,
    "pre_space_delay": 0.1,
    "next_pattern_delay": 3.0,
    "idle_loop_delay": 0.25,
}

# Define your templates and their labels
templates = {
    "BLUE_UP": {"template": cv2.imread("images/blue/up.png"), "key": "W"},
    "BLUE_DOWN": {
        "template": cv2.imread("images/blue/down.png"),
        "key": "S",
    },
    "BLUE_LEFT": {"template": cv2.imread("images/blue/left.png"), "key": "A"},
    "BLUE_RIGHT": {"template": cv2.imread("images/blue/right.png"), "key": "D"},
    "RED_UP": {"template": cv2.imread("images/red/up.png"), "key": "S"},
    "RED_DOWN": {"template": cv2.imread("images/red/down.png"), "key": "W"},
    "RED_LEFT": {"template": cv2.imread("images/red/left.png"), "key": "D"},
    "RED_RIGHT": {"template": cv2.imread("images/red/right.png"), "key": "A"},
}


def capture_region(region_name: str):
    print(f"\nSet {region_name} region:")
    print("1) Move mouse to UPPER-LEFT corner, then press F8...")
    wait_for_capture_key()
    upper_left = pyautogui.position()
    print(f"Saved upper-left: ({upper_left.x}, {upper_left.y})")

    print("2) Move mouse to LOWER-RIGHT corner, then press F8...")
    wait_for_capture_key()
    lower_right = pyautogui.position()
    print(f"Saved lower-right: ({lower_right.x}, {lower_right.y})")

    width = lower_right.x - upper_left.x
    height = lower_right.y - upper_left.y

    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid region for {region_name}: lower-right must be below and to the right of upper-left."
        )

    return (upper_left.x, upper_left.y, width, height)


def wait_for_capture_key() -> None:
    def on_press(key):
        if key == pynput_keyboard.Key.f8:
            return False

    with pynput_keyboard.Listener(on_press=on_press) as listener:
        listener.join()


def load_config() -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)
    return config


def save_config(config: configparser.ConfigParser) -> None:
    with open(CONFIG_FILE, "w", encoding="utf-8") as config_file:
        config.write(config_file)


def parse_region(
    config: configparser.ConfigParser, section_name: str
) -> tuple[int, int, int, int] | None:
    if section_name not in config:
        return None

    try:
        x = config.getint(section_name, "x")
        y = config.getint(section_name, "y")
        width = config.getint(section_name, "width")
        height = config.getint(section_name, "height")
    except (ValueError, configparser.Error):
        return None

    if width <= 0 or height <= 0:
        return None

    return (x, y, width, height)


def save_region(
    config: configparser.ConfigParser,
    section_name: str,
    region: tuple[int, int, int, int],
) -> None:
    x, y, width, height = region
    config[section_name] = {
        "x": str(x),
        "y": str(y),
        "width": str(width),
        "height": str(height),
    }


def load_delays(config: configparser.ConfigParser) -> dict[str, float]:
    if "delays" not in config:
        config["delays"] = {}

    delays = {}
    for key, default_value in DEFAULT_DELAYS.items():
        try:
            delays[key] = config.getfloat("delays", key)
        except (ValueError, configparser.Error):
            delays[key] = default_value

        config["delays"][key] = str(delays[key])

    return delays


def setup_regions_and_delays() -> (
    tuple[tuple[int, int, int, int], tuple[int, int, int, int], dict[str, float]]
):
    config = load_config()
    config_changed = False
    had_delays_section = "delays" in config
    existing_delay_keys = set(config["delays"].keys()) if had_delays_section else set()

    game_coords = parse_region(config, "game_region")
    if game_coords is None:
        print("Game region is not configured. Capture required.")
        game_coords = capture_region("game")
        save_region(config, "game_region", game_coords)
        config_changed = True
    else:
        print(f"Loaded game region from {CONFIG_FILE}: {game_coords}")

    timing_coords = parse_region(config, "timing_region")
    if timing_coords is None:
        print("Timing region is not configured. Capture required.")
        timing_coords = capture_region("timing")
        save_region(config, "timing_region", timing_coords)
        config_changed = True
    else:
        print(f"Loaded timing region from {CONFIG_FILE}: {timing_coords}")

    delays = load_delays(config)
    if not had_delays_section:
        config_changed = True
    else:
        for key in DEFAULT_DELAYS:
            if key not in existing_delay_keys:
                config_changed = True
                break

    if config_changed:
        save_config(config)
        print(f"Saved settings to {CONFIG_FILE}")

    return game_coords, timing_coords, delays


def get_color_pattern(image: Image):
    # screen_image should be the BGR image from cv2.imread() or a screenshot
    img_color = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    found_arrows = []

    for label, template_info in templates.items():
        template = template_info["template"]

        # w, h = template.shape[1], template.shape[0]

        # We match against the color image (img_color) instead of img_gray
        res = cv2.matchTemplate(img_color, template, cv2.TM_CCOEFF_NORMED)

        # High threshold (0.8+) ensures color accuracy
        threshold = 0.7
        loc = np.where(res >= threshold)

        for pt in zip(*loc[::-1]):
            # Duplicate filter (checks if we already found an arrow at this X)
            if not any(abs(pt[0] - existing["x"]) < 15 for existing in found_arrows):
                found_arrows.append({"x": pt[0], "label": label})

    found_arrows.sort(key=lambda x: x["x"])
    return [arrow["label"] for arrow in found_arrows]


def detect_space_pattern(image: Image):
    space_template = cv2.imread("images/perfect.png")
    img_color = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    res = cv2.matchTemplate(img_color, space_template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.7
    loc = np.where(res >= threshold)
    return len(loc[0]) > 0  # Returns True if space pattern is detected


def main() -> None:
    print("Coordinate setup starting. Keep the game visible.")
    game_coords, timing_coords, delays = setup_regions_and_delays()
    print(f"\nGame region: {game_coords}")
    print(f"Timing region: {timing_coords}\n")
    print(
        "Delays: "
        f"key_press_delay={delays['key_press_delay']}s, "
        f"pre_space_delay={delays['pre_space_delay']}s, "
        f"next_pattern_delay={delays['next_pattern_delay']}s, "
        f"idle_loop_delay={delays['idle_loop_delay']}s"
    )

    while True:
        # Check for the image every 1 second
        image = pyautogui.screenshot(region=game_coords)
        # image.show()
        pattern = get_color_pattern(image)

        if not pattern:
            print("No arrows detected.")
        else:
            print(f"Detected Pattern: {' -> '.join(pattern)}")
            for arrow in pattern:
                key_to_press = templates[arrow]["key"]
                print(f"Pressing {key_to_press} for {arrow}")
                time.sleep(delays["key_press_delay"])
                pyautogui.press(key_to_press)
            while True:
                image = pyautogui.screenshot(region=timing_coords)
                space_detected = detect_space_pattern(image)
                if space_detected:
                    print("Space pattern detected.")
                    time.sleep(delays["pre_space_delay"])
                    pyautogui.press("SPACE")
                    break
            time.sleep(delays["next_pattern_delay"])

        time.sleep(delays["idle_loop_delay"])


main()
