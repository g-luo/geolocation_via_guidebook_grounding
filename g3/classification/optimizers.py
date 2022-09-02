import logging

def get_optimizer_parameters(config, module):
    lr = config["params"]["lr"]
    finetune_lr_multiplier = config["params"].get("finetune_lr_multiplier", 1)
    logging.info(f"Using lr {lr} finetune_lr_multiplier {finetune_lr_multiplier}")
    parameters = []
    for name, submodule in module.named_children():
        submodule_parameters = config["params"]
        if name.startswith("attn_layer"):
            submodule_parameters["lr"] = config["params"]["attention_lr"]
        elif name.startswith("classifier") or name.startswith("beta"):
            submodule_parameters["lr"] = lr
        else:
            submodule_parameters["lr"] = lr * finetune_lr_multiplier
            if finetune_lr_multiplier == 0:
                for p in submodule.parameters():
                    p.requires_grad = False
        submodule_parameters = {**{"params": submodule.parameters()}, **submodule_parameters}
        parameters.append(submodule_parameters)
    return parameters
