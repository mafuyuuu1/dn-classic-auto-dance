from PIL import Image
import pyautogui
import time

import cv2
import numpy as np

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
    game_coords = (1900, 866, 2933 - 1900, 951 - 866)
    space_coords = (2653, 813, 2770 - 2653, 865 - 813)
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
                pyautogui.press(key_to_press)
                time.sleep(0.1)  # Small delay between key presses
            while True:
                image = pyautogui.screenshot(region=space_coords)
                space_detected = detect_space_pattern(image)
                if space_detected:
                    print("Space pattern detected.")
                    pyautogui.press("SPACE")
                    time.sleep(0.1)  # Small delay after pressing space
                    break
            time.sleep(3)  # Wait a bit before checking for the next pattern

        time.sleep(0.25)


main()
