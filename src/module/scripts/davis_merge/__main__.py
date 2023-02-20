import argparse
import logging
import sys

from module.scripts.davis_merge.merge_script import create_davis_video


logger = logging.getLogger()
str_handler = logging.StreamHandler(sys.stdout)
fmt = logging.Formatter(
    "[%(asctime)s\t%(levelname)s\t%(name)s]: \n%(message)s",
)
logger.setLevel(logging.INFO)
str_handler.setFormatter(fmt)
logger.addHandler(str_handler)


def parse_args():
    parser = argparse.ArgumentParser("davis")
    parser.add_argument(
        "images",
        help="folder with subfolders containing images",
    )
    parser.add_argument(
        "annotations",
        help="folder with subfolders containing annotations",
    )
    parser.add_argument(
        "savepath",
        help="path to save",
    )
    parser.add_argument(
        "name",
        help="name of final video",
    )
    parser.add_argument(
        "--wks",
        required=False,
        type=int,
        help="number of subprocess for parallel image annotation",
        default=1,
    )
    parser.add_argument(
        "--fr",
        required=False,
        type=int,
        help="framerate",
        default=15,
    )
    parser.add_argument(
        "--alpha",
        required=False,
        type=float,
        help="transperancy factor for annotation",
        default=0.5,
    )
    parser.add_argument(
        "--thck",
        required=False,
        type=int,
        help="annotation border thickness",
        default=1,
    )

    return parser.parse_args()


def main():
    args = parse_args()
    create_davis_video(
        images=args.images,
        annotations=args.annotations,
        path_to_save=args.savepath,
        name=args.name,
        n_workers=args.wks,
        framerate=args.fr,
        alpha=args.alpha,
        border_thickness=args.thck,
    )


if __name__ == "__main__":
    main()