import yaml


DEFAULTS = {
    # random seed for reproducibility, a large number is preferred
    "init_rand_seed": 1234567891,
    "devices": ['cuda:4'], # default: single gpu
    "loader": {
        "batch_size": 8,
        "num_workers": 4,
    },
    # optimizer (for training)
    "opt": {
        "type": "AdamW", # solver: SGD or AdamW
        "weight_decay": 0.05,
        "learning_rate": 1e-4,
        "epochs": 30,  # excluding the warmup epochs
        # lr scheduler: cosine / multistep
        "warmup": True,
        "warmup_epochs": 5,
        "schedule_type": "cosine",
    }
}

def _merge(src, dst):
    for k, v in src.items():
        if k in dst:
            if isinstance(v, dict):
                _merge(src[k], dst[k])
        else:
            dst[k] = v

def load_default_config():
    config = DEFAULTS
    return config

def _update_config(config):
    # Common transfers
    if "dataset" in config:
        # Pass generic dataset info to model config if needed
        if "input_dim" in config["dataset"]:
            input_dim = config["dataset"]["input_dim"]

            # Auto-multiply input_dim when concat_views is enabled
            if config["dataset"]["concat_views"]:
                egoexo_type = config["dataset"]["egoexo_type"]
                if egoexo_type == "exo":
                    input_dim = input_dim * 4   # exo1 + exo2 + exo3 + exo4
                elif egoexo_type == "both":
                    input_dim = input_dim * 5   # ego + exo1 + exo2 + exo3 + exo4

            config["model"]["input_dim"] = input_dim
        if "motion_window_size" in config["dataset"]:
            config["model"]["motion_window_size"] = config["dataset"]["motion_window_size"]
        if "num_classes" in config["dataset"]:
            config["model"]["num_classes"] = config["dataset"]["num_classes"]
        if "max_seq_len" in config["dataset"]:
            config["model"]["max_seq_len"] = config["dataset"]["max_seq_len"]
        if "fps_out" in config["dataset"]:
            config["model"]["fps_out"] = config["dataset"]["fps_out"]
            config["loader"]["fps_out"] = config["dataset"]["fps_out"]
        if "egoexo_type" in config["dataset"]:
            config["evaluator"]["egoexo_type"] = config["dataset"]["egoexo_type"]
        if "concat_views" in config["dataset"]:
            config["evaluator"]["concat_views"] = config["dataset"]["concat_views"]
            # Evaluator needs feat info to filter takes with missing features
            if config["dataset"].get("concat_views", False):
                config["evaluator"]["feat_folder"] = config["dataset"]["feat_folder"]
                config["evaluator"]["file_prefix"] = config["dataset"].get("file_prefix") or ""
                config["evaluator"]["file_ext"] = config["dataset"].get("file_ext", ".npy")
        # cross_view uses the same take_uid grouping as concat_views for evaluation
        if config["dataset"].get("cross_view", False):
            config["evaluator"]["concat_views"] = True
            config["evaluator"]["feat_folder"] = config["dataset"]["feat_folder"]
            config["evaluator"]["file_prefix"] = config["dataset"].get("file_prefix") or ""
            config["evaluator"]["file_ext"] = config["dataset"].get("file_ext", ".npy")
        # Pass cross_view flag to model so it can skip building unused modules
        if "cross_view" in config["dataset"]:
            config["model"]["build_crossview"] = config["dataset"]["cross_view"]
            # config["model"]["build_crossview"] = True

        # if "text_anchors_file" in config["dataset"]:
        #     config["model"]["text_anchors_file"] = config["dataset"]["text_anchors_file"]

    if "evaluator" in config:
        config["evaluator"]["dataset_name"] = config["dataset_name"]

    if "train_cfg" in config:
        config["model"]["train_cfg"] = config["train_cfg"]

    if "test_cfg" in config:
        config["model"]["test_cfg"] = config["test_cfg"]

    return config

def load_config(config_file, defaults=DEFAULTS):
    with open(config_file, "r") as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    _merge(defaults, config)
    config = _update_config(config)
    return config