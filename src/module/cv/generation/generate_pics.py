import cv2
from pathlib import Path
from tqdm import tqdm
from numpy.random import default_rng

from module.cv.utils.constructor import generate_pic
from module.cv.utils.img_data import GenParams

NUM_OF_GENERATIONS = 1000
TEST_NUM_OF_GEN = 300

CLASS_COLORS = {
    "golden_apple": (208, 46, 46),
    "green_apple": (0, 204, 102),
    "red_apple": (204, 204, 0),
    "red_green_apple": (0, 102, 204),
    "red_yellow_apple": (102, 0, 204),
    "striped_apple": (96, 96, 96),
}

CLASS_COMP = {
    "red_green_apple": "Apple A",
    "red_apple": "Apple B",
    "striped_apple": "Apple C",
    "green_apple": "Apple D",
    "red_yellow_apple": "Apple E",
    "golden_apple": "Apple F",
}

MAIN_PARAMS = GenParams(
    generator=default_rng(seed=42),
    h_limits=(258, 322),  # shapes existing in original dataset
    w_limits=(320, 480),  # shapes existing in original dataset
    prim_limit=7,
    size_factor=(0.2, 0.3),
    rotate=(-90, 90),
    gaus_blur=(3, 3),
)

TEST_PARAMS = MAIN_PARAMS
TEST_PARAMS.generator = default_rng(seed=43)

BG = Path(__file__).parent / "back"
BGS_P = []

for bgp in BG.iterdir():
    BGS_P.append(str(bgp))

PRIMS = Path(__file__).parent / "prim_v2"
SAVE_PATH = Path(__file__).parent / "generated_data_v2"
TEST_SAVE_PATH = Path(__file__).parent / "generated_test_data_v2"

def generate_one_class_pics(save_path: Path, class_name: str, gen_params: GenParams, num: int = 1000) -> None:
    save_pcls = save_path / CLASS_COMP[class_name]
    annot_path = save_pcls / "annot"
    annot_path.mkdir(parents=True, exist_ok=True)
    pics_path = save_pcls / "images"
    pics_path.mkdir(parents=True, exist_ok=True)

    pcl = PRIMS / class_name

    for i in tqdm(range(num), desc=f"Generating {class_name}"):
        pic, annot = generate_pic(
            classes_path=[str(pcl)],
            back_paths=BGS_P,
            cls_colors=CLASS_COLORS,
            gen_params=gen_params,
        )

        cv2.imwrite(str(annot_path / f"apple_{i}.png"), annot)
        cv2.imwrite(str(pics_path / f"apple_{i}.jpeg"), pic)


if __name__ == "__main__":
    generate_one_class_pics(SAVE_PATH, "red_green_apple", MAIN_PARAMS, NUM_OF_GENERATIONS)
    generate_one_class_pics(SAVE_PATH, "red_apple", MAIN_PARAMS, NUM_OF_GENERATIONS)
    generate_one_class_pics(SAVE_PATH, "striped_apple", MAIN_PARAMS, NUM_OF_GENERATIONS)
    generate_one_class_pics(SAVE_PATH, "green_apple", MAIN_PARAMS, NUM_OF_GENERATIONS)
    generate_one_class_pics(SAVE_PATH, "red_yellow_apple", MAIN_PARAMS, NUM_OF_GENERATIONS)
    generate_one_class_pics(SAVE_PATH, "golden_apple", MAIN_PARAMS, NUM_OF_GENERATIONS)

    generate_one_class_pics(TEST_SAVE_PATH, "red_green_apple", TEST_PARAMS, TEST_NUM_OF_GEN)
    generate_one_class_pics(TEST_SAVE_PATH, "red_apple", TEST_PARAMS, TEST_NUM_OF_GEN)
    generate_one_class_pics(TEST_SAVE_PATH, "striped_apple", TEST_PARAMS, TEST_NUM_OF_GEN)
    generate_one_class_pics(TEST_SAVE_PATH, "green_apple", TEST_PARAMS, TEST_NUM_OF_GEN)
    generate_one_class_pics(TEST_SAVE_PATH, "red_yellow_apple", TEST_PARAMS, TEST_NUM_OF_GEN)
    generate_one_class_pics(TEST_SAVE_PATH, "golden_apple", TEST_PARAMS, TEST_NUM_OF_GEN)
        