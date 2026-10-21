from ..models.s2no_pretrain import S2NO_pretrain
from ..models.bfno_pretrain import BFNO_pretrain
from ..models.fno_pretrain import FNO_pretrain
from ..models.unet_pretrain import UNet_pretrain


def model_select(c_nn, c_train, c_data):
    name = c_nn["name"]

    common_paths = dict(
        val_exp=c_nn.get("val_exp", False),
        speed_path_breast=c_data.get("speed_path_breast", ""),
        speed_path_arm=c_data.get("speed_path_arm", ""),
        speed_path_limb=c_data.get("speed_path_limb", ""),
        valid_num_breast=c_data["valid_num_breast"],
        valid_num_arm=c_data["valid_num_arm"],
        src_path_breast=c_data["src_path_breast"],
        src_path_arm=c_data["src_path_arm"],
        gt_path_breast=c_data["gt_path_breast"],
        gt_path_arm=c_data["gt_path_arm"],
        src_path_limb=c_data["src_path_limb"],
        gt_path_limb=c_data["gt_path_limb"],
    )

    if name == "S2NO_pretrain":
        return S2NO_pretrain(
            width=c_nn["features"],
            modes1=c_nn["wavenumber"],
            modes2=c_nn["wavenumber"],
            layer_num=c_nn["layer_num"],
            padding=c_nn["padding"],
            dim_input=1,
            loss=c_nn["loss"],
            learning_rate=c_train["lr"],
            step_size=c_train["step_size"],
            gamma=c_train["gamma"],
            weight_decay=c_train["weight_decay"],
            F_feature=c_train["F_feature"],
            add_term=c_train["add_term"],
            eta_min=c_train["eta_min"],
            source_type=c_nn["source_type"],
            **common_paths,
        )

    if name == "BFNO_pretrain":
        return BFNO_pretrain(
            width=c_nn["features"],
            modes1=c_nn["wavenumber"],
            modes2=c_nn["wavenumber"],
            layer_num=c_nn["layer_num"],
            padding=c_nn["padding"],
            dim_input=1,
            source_type=c_nn["source_type"],
            loss=c_nn["loss"],
            learning_rate=c_train["lr"],
            step_size=c_train["step_size"],
            gamma=c_train["gamma"],
            weight_decay=c_train["weight_decay"],
            F_feature=c_train["F_feature"],
            add_term=c_train["add_term"],
            eta_min=c_train["eta_min"],
            **common_paths,
        )

    if name == "FNO_pretrain":
        return FNO_pretrain(
            wavenumber=c_nn["wavenumber"],
            features_=c_nn["features"],
            padding=c_nn["padding"],
            activation="relu",
            lifting=None,
            proj=None,
            dim_input=1,
            source_type=c_nn["source_type"],
            loss=c_nn["loss"],
            learning_rate=c_train["lr"],
            step_size=c_train["step_size"],
            gamma=c_train["gamma"],
            weight_decay=c_train["weight_decay"],
            add_term=c_train["add_term"],
            eta_min=c_train["eta_min"],
            **common_paths,
        )

    if name == "UNet_pretrain":
        return UNet_pretrain(
            in_ch=c_nn["in_ch"],
            out_ch=c_nn["out_ch"],
            scales=c_nn["scales"],
            skip=c_nn["skip"],
            channels=c_nn["channels"],
            use_sigmoid=False,
            use_norm=True,
            learning_rate=c_train["lr"],
            step_size=c_train["step_size"],
            gamma=c_train["gamma"],
            weight_decay=c_train["weight_decay"],
            loss=c_nn["loss"],
            F_feature=c_train["F_feature"],
            add_term=c_train["add_term"],
            eta_min=c_train["eta_min"],
            source_type=c_nn["source_type"],
            **common_paths,
        )

    raise ValueError(f"Unsupported model name: {name}")
