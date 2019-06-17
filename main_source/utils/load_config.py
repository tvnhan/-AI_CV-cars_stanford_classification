import configparser
import ast


def _getConfigValue(func, section, option, default_value, is_check_required=False):
    try:
        value = func(section, option)
    except:
        value = default_value

    if is_check_required and value is None:
        raise ValueError("PLEASE FILL THIS FIELD")
    return value


def _getConfigValueInDictioary(func, section, option, default_value, is_check_required=False):
    try:
        value = ast.literal_eval(func(section, option))
    except:
        print('Canot read value')
        value = default_value

    if is_check_required and value is None:
        raise ValueError("PLEASE FILL THIS FIELD")
    return value


def load_config(path):
    Config = configparser.ConfigParser()
    Config.read(path)
    dict_config = {}

    mainSection = "MAIN"
    dict_config["approach"] = _getConfigValue(Config.get, mainSection, "approach", None, is_check_required=True)
    dict_config["model"] = _getConfigValue(Config.get, mainSection, "model", None)

    trainSection = "TRAIN"
    dict_config["resume_pretrained"] = _getConfigValue(Config.getboolean, trainSection, "resume_pretrained", False)
    dict_config["link_pretrained"] = _getConfigValue(Config.get, trainSection, "link_pretrained", None)

    dict_config["num_classes"] = _getConfigValue(Config.getint, trainSection, "num_classes", 5)
    dict_config["size_width"] = _getConfigValue(Config.getint, trainSection, "size_width", 5)
    dict_config["size_height"] = _getConfigValue(Config.getint, trainSection, "size_height", 5)

    dict_config["proposal_num"] = _getConfigValue(Config.getint, trainSection, "proposal_num", 5)
    dict_config["cat_num"] = _getConfigValue(Config.getint, trainSection, "cat_num", 4)
    dict_config["batch_size"] = _getConfigValue(Config.getint, trainSection, "batch_size", 16)
    dict_config["lr"] = _getConfigValue(Config.getfloat, trainSection, "learning_rate", 0.001)
    dict_config["wd"] = _getConfigValue(Config.getfloat, trainSection, "weight_decay", 1e-4)

    dict_config["save_model_dir"] = _getConfigValue(Config.get, trainSection, "save_model_dir", "./outputs")
    dict_config["save_checkpoint_freq"] = _getConfigValue(Config.getint, trainSection, "save_checkpoint_freq", 2)

    dict_config["start_epoch"] = _getConfigValue(Config.getint, trainSection, "start_epoch", 0)
    dict_config["epochs"] = _getConfigValue(Config.getint, trainSection, "epochs", 300)

    dict_config["link_train_csv"] = _getConfigValue(Config.get, trainSection, "link_train_csv", "")
    dict_config["root_folder_train"] = _getConfigValue(Config.get, trainSection, "root_folder_train", "./dataset/train")
    dict_config["link_val_csv"] = _getConfigValue(Config.get, trainSection, "link_val_csv", "")
    dict_config["root_folder_val"] = _getConfigValue(Config.get, trainSection, "root_folder_val", "./dataset/val")
    return dict_config
