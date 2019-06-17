# Import Libraries
import argparse
import os
import sys
sys.path.append('../')
import main_source.utils.load_config as load_config



### parse arguments
parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('config', metavar='DIR', help='path to config file')


def main():
    # Loading config
    global dict_config
    args = parser.parse_args()
    dict_config = load_config.load_config(args.config)
    print(dict_config)

    # Process use GPU or CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    ### Choose an approach for problems
    switcher = {
        "NTS": launch_NTS,
    }
    appr = switcher.get(dict_config["approach"])

    ### Run
    return appr()

def launch_NTS():
    import main_source.approaches.modified_NTS.approach_nts as apr
    return apr.approach_process(dict_config)

if __name__ == '__main__':
    main()
