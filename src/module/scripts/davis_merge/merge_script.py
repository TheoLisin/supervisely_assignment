import os
import cv2
from itertools import chain
from typing import List, NamedTuple
from logging import getLogger
from pathlib import Path
from multiprocessing import Pool

from module.scripts.davis_merge.annotation import annotate_image_sequence, PathT
from module.scripts.davis_merge.videomaker import sequence_to_video

logger = getLogger(__name__)


class WorkerArgs(NamedTuple):
    folder_name: str
    images: Path
    annotations: Path
    alpha: float
    border_thickness: int


def create_davis_video(
    images: PathT,
    annotations: PathT,
    path_to_save: PathT,
    name: str,
    n_workers: int = 1,
    framerate: float = 15.0,
    alpha: float = 0.5,
    border_thickness: int = 1,
) -> None:
    """Stuck all DAVIS frames to one video with annotation.


    Args:
        images (PathT): folder with subfolders containing images.
        annotations (PathT): folder with subfolders containing annotations.
        path_to_save (PathT): path to save video
        name (str): name of created video
        n_workers (int): Number of subprocess for parallel image annotation.
            Defaults to 1.
        framerate (float): Defaults to 15.0.
        alpha (float): Transperancy factor for annotation. Defaults to 0.5
        border_thickness (int): annotation border thickness. Defaults to 1 px.
    """
    images = Path(images)
    annotations = Path(annotations)

    if n_workers <= 0:
        n_workers = 1
        logger.warning("n_workers <= 0, set it to default 1.")

    folder_names = os.listdir(images)

    args = [
        WorkerArgs(
            name,
            images,
            annotations,
            alpha,
            border_thickness,
        )
        for name in folder_names
    ]

    with Pool(processes=n_workers) as pool:
        fs = pool.map(_subprocess_sequence_builder, args)

    shapes = [frames[0].shape for frames in fs]

    height = max([shape[0] for shape in shapes])
    width = max([shape[1] for shape in shapes])
    size = (height, width, 3)

    final_seq = chain(*fs)
    sequence_to_video(final_seq, path_to_save, name, size=size, framerate=framerate)


def _subprocess_sequence_builder(args: WorkerArgs) -> List[cv2.Mat]:
    logger.info(f"Worker start '{args.folder_name}'.")
    imgs = args.images / args.folder_name
    annots = args.annotations / args.folder_name
    img_seq = annotate_image_sequence(imgs, annots, args.alpha, args.border_thickness)

    logger.info("Worker finished adding annotations.")
    return img_seq
