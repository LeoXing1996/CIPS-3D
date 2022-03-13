import os
import sys

from tl2 import tl2_utils
from tl2.launch.launch_utils import \
    (setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)

from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--n-gpus', type=int, default=1,
                        help='How many gpu to use, defaults only use 1 gpu to debug.')
    parser.add_argument('--is-high', action='store_true',
                        help='Whether use the high resolution config.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = get_args()
    tl_command = 'train_ffhq_high' if args.is_high else 'train_ffhq'

    tl_opts_list = tl2_utils.parser_args_from_list(
        name="--tl_opts", argv_list=sys.argv, type='list')
    tl_opts = ' '.join(tl_opts_list)
    print(f'tl_opts:\n {tl_opts}')

    outdir = './work_dir/ffhq'
    resume = os.path.isdir(f"{outdir}/ckptdir/resume") and \
        tl2_utils.parser_args_from_list(
            name="--tl_outdir", argv_list=sys.argv, type='str') is not None
    argv_str = f"""
                --tl_config_file exp/cips3d/configs/ffhq_exp.yaml
                --tl_command {tl_command}
                --tl_outdir {outdir}
                --tl_opts {tl_opts}
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)

    n_gpus = args.n_gpus
    PORT = os.environ.get('PORT', 12345)

    os.environ['DNNLIB_CACHE_DIR'] = "cache_dnnlib"
    os.environ['TORCH_EXTENSIONS_DIR'] = "cache_torch_extensions"
    os.environ['PATH'] = f"{os.path.dirname(sys.executable)}:{os.environ['PATH']}"
    os.environ['MAX_JOBS '] = "8"

    cmd_str = f"""
        python train.py
        --port {PORT}

        {get_append_cmd_str(args)}
        """
    cmd_str += f"""
                --tl_opts num_workers {n_gpus}
                {tl_opts}
                """
    start_cmd_run(cmd_str)
