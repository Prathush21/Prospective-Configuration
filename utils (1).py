import predictive_coding as pc
import torch


def create_model(predictive_coding, acf, model_type_order, cnn_layers, linear_layers, batch_norm_layers=None, dropout_layers=None, max_pool_layers=None, loss_fn=''):

    model_type_order = eval(model_type_order)
    model = []

    # Add CNN layers
    for cnn_key, cnn_layer in cnn_layers.items():
        for model_type in model_type_order:
            if model_type == 'Weights':
                model_ = eval(cnn_layer['fn'])(**cnn_layer['kwargs'])
                model.append(model_)
            elif model_type == 'Acf':
                model_ = eval(acf)()
                model.append(model_)
            elif model_type == 'PCLayer':
                model_ = pc.PCLayer()
                model.append(model_)
            elif model_type == 'Pool' and max_pool_layers:
                pool_key = f"max_pool_{cnn_key.split('_')[-1]}"
                if pool_key in max_pool_layers:
                    pool_layer = max_pool_layers[pool_key]
                    model_ = eval(pool_layer['fn'])(**pool_layer['kwargs'])
                    model.append(model_)
            elif model_type == 'BatchNorm' and batch_norm_layers:
                bn_key = f"batch_norm_{cnn_key.split('_')[-1]}"
                if bn_key in batch_norm_layers:
                    bn_layer = batch_norm_layers[bn_key]
                    model_ = eval(bn_layer['fn'])(**bn_layer['kwargs'])
                    model.append(model_)
            elif model_type == 'Dropout' and dropout_layers:
                dropout_key = f"dropout_{cnn_key.split('_')[-1]}"
                if dropout_key in dropout_layers:
                    dropout_layer = dropout_layers[dropout_key]
                    model_ = eval(dropout_layer['fn'])(**dropout_layer['kwargs'])
                    model.append(model_)
            else:
                raise ValueError(f'Model type {model_type} not found')
    
    # Add flatten layer
    model.append(torch.nn.Flatten())
    
    # Add linear layers
    for linear_key, linear_layer in linear_layers.items():
        if linear_key == 'last':
            model_ = eval(linear_layer['fn'])(**linear_layer['kwargs'])
            model.append(model_)
        else:
            for model_type in model_type_order:
                if model_type == 'Weights':
                    model_ = eval(linear_layer['fn'])(**linear_layer['kwargs'])
                    model.append(model_)
                elif model_type == 'Acf':
                    model_ = eval(acf)()
                    model.append(model_)
                elif model_type == 'PCLayer':
                    model_ = pc.PCLayer()
                    model.append(model_)
                elif model_type == 'Dropout' and dropout_layers:
                    dropout_key = f"dropout_{linear_key.split('_')[-1]}"
                    if dropout_key in dropout_layers:
                        dropout_layer = dropout_layers[dropout_key]
                        model_ = eval(dropout_layer['fn'])(**dropout_layer['kwargs'])
                        model.append(model_)
    
    if loss_fn == 'cross_entropy':
        model.append(torch.nn.Softmax(dim=1))
    
    # Remove predictive coding layers if not needed
    if not predictive_coding:
        model = [layer for layer in model if not isinstance(layer, pc.PCLayer)]
    
    # Create sequential model
    model = torch.nn.Sequential(*model)
    
    return model
