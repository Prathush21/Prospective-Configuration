import predictive_coding as pc
import torch
from torchvision import models


def create_model(
    predictive_coding, acf, model_type_order, cnn_layers, linear_layers, loss_fn=""
):
    
    # Load the pre-trained AlexNet model
    alexnet = models.alexnet(pretrained=True)
    # Map your model's layers to AlexNet's layers
    alexnet_layers = list(alexnet.features)
    alexnet_index = 0

    model_type_order = eval(model_type_order)

    model = []

    for cnn_key, cnn_layer in cnn_layers.items():
        for model_type in model_type_order:
            if model_type == "Weights":
                model_ = eval(cnn_layer["fn"])(**cnn_layer["kwargs"])
                # Initialize weights with those of pre-trained AlexNet
                if isinstance(model_, torch.nn.Conv2d):
                    while not isinstance(alexnet_layers[alexnet_index], torch.nn.Conv2d):
                        alexnet_index += 1
                    model_.weight.data = alexnet_layers[alexnet_index].weight.data.clone()
                    if model_.bias is not None:
                        model_.bias.data = alexnet_layers[alexnet_index].bias.data.clone()
                    # Freeze the convolutional layer
                    model_.weight.requires_grad = False
                    if model_.bias is not None:
                        model_.bias.requires_grad = False
                    alexnet_index += 1
            elif model_type == "Acf":
                model_ = eval(acf)()
            elif model_type == "PCLayer":
                model_ = pc.PCLayer()
            elif model_type == "Pool":
                if cnn_key not in  ["cnn_2", "cnn_3"]:
                    model_ = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
            elif model_type == "Dropout":
                continue
            else:
                raise ValueError("model_type not found")
            model.append(model_)

    model.append(torch.nn.Flatten())

    for linear_key, linear_layer in linear_layers.items():
        if linear_key == "last":
            model_ = eval(linear_layer["fn"])(**linear_layer["kwargs"])
            model.append(model_)
        else:
            for model_type in model_type_order:
                if model_type == "Weights":
                    model_ = eval(linear_layer["fn"])(**linear_layer["kwargs"])
                elif model_type == "Acf":
                    model_ = eval(acf)()
                elif model_type == "PCLayer":
                    model_ = pc.PCLayer()
                elif model_type == 'Dropout':
                    model_ = torch.nn.Dropout()
                model.append(model_)

    if loss_fn == "cross_entropy":
        model.append(torch.nn.Softmax())

    # decide pc_layer
    for model_ in model:
        if isinstance(model_, pc.PCLayer):
            if not predictive_coding:
                model.remove(model_)

    # # initialize
    # for model_ in model:
    #     if isinstance(model_, torch.nn.Linear):
    #         eval(init_fn)(
    #             model_.weight,
    #             **init_fn_kwarg,
    #         )

    # create sequential
    model = torch.nn.Sequential(*model)

    return model
