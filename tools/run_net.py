import argparse
import jittor as jt
from jseg.runner import TrainRunner, EvalRunner, TestRunner
from jseg.config import init_cfg
from jseg.config.config import update_cfg

jt.cudnn.set_max_workspace_ratio(0.0)


def main():
    parser = argparse.ArgumentParser(
        description="Jittor Semantic segmentation Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--task",
        default="train",
        help="train,val test",
        type=str,
    )

    parser.add_argument(
        "--resume",
        default=None,
        help="resume path",
        type=str,
    )
    parser.add_argument(
        "--save-dir",
        default="./results",
        type=str,
    )

    parser.add_argument("--no_cuda", action='store_true')

    parser.add_argument("--efficient_val", action='store_true')
    args = parser.parse_args()

    if not args.no_cuda:
        jt.flags.use_cuda = 1

    assert args.task in [
        "train", "val", "test"
    ], f"{args.task} not support, please choose [train,val,test]"

    if args.config_file:
        init_cfg(args.config_file)

    if args.resume:
        update_cfg(resume_path=args.resume)
    if args.efficient_val:
        update_cfg(efficient_val=args.efficient_val)

    runner = None

    if args.task == "train":
        runner = TrainRunner()
    elif args.task == "val":
        runner = EvalRunner()
    elif args.task == "test":
        runner = TestRunner(save_dir=args.save_dir)

    runner.run()


if __name__ == "__main__":
    main()
