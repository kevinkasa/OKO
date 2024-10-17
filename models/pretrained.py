import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core import FrozenDict, frozen_dict
from jax_resnet import pretrained_resnet, slice_variables, Sequential

from models.oko_head import OKOHead


class Model(nn.Module):
    '''Combines backbone and head model'''
    backbone: Sequential
    num_classes: int
    k: int

    def setup(self) -> None:
        '''
        Can choose from OKO training or regular CE training by specifiying value of k
        '''
        if self.k > 0:
            print('Using OKO Head')
            self.head = OKOHead(
                backbone="resnet",
                num_classes=self.num_classes,
                k=self.k,
            )
        elif self.k == 0:
            print('Using regular Dense Head')
            self.head = nn.Dense(self.num_classes, name="clf_head")
        else:
            raise ValueError(f'K needs to be >=0, currently: {self.k}')

    def __call__(self, inputs, train: bool = True):
        x = self.backbone(inputs)
        # average pool layer
        x = jnp.mean(x, axis=(1, 2))
        x = self.head(x, train) if self.k > 0 else self.head(x)
        return x


def _get_backbone_and_params(model_arch: str):
    '''
    Get backbone and params
    1. Loads pretrained model (resnet18)
    2. Get model and param structure except last 2 layers
    3. Extract the corresponding subset of the variables dict
    INPUT : model_arch
    RETURNS backbone , backbone_params
    '''
    if model_arch == 'resnet18':
        resnet_tmpl, params = pretrained_resnet(18)
        model = resnet_tmpl()
    elif model_arch == 'resnet50':
        resnet_tmpl, params = pretrained_resnet(50)
        model = resnet_tmpl()
    else:
        raise NotImplementedError

    # get model & param structure for backbone
    start, end = 0, len(model.layers) - 2
    backbone = Sequential(model.layers[start:end])
    backbone_params = slice_variables(params, start, end)
    return backbone, backbone_params


def get_model_and_variables(model_arch: str, head_init_key: int, num_classes: int, k: int):
    '''
    Get model and variables
    1. Initialize inputs(shape=(1, 224, 224, 3))
    2. Get backbone and params
    3. Create Model with backbone, num_classes, k
    4. Initialize Model's variables
    5. Merge pretrained backbone parameters with the model's variables

    INPUT:
        - model_arch: str, architecture of the backbone (e.g., 'resnet18')
        - head_init_key: int, seed for head initialization
        - num_classes: int, number of output classes
        - k: int, controls head type (k > 0 for OKOHead, k = 0 for nn.Dense)

    RETURNS:
        - model: Instance of Model
        - variables: FrozenDict containing model parameters and batch statistics
    '''

    # Step 1: Initialize dummy inputs
    if k == 0:
        inputs = jnp.ones((1, 224, 224, 3), jnp.float32)
    else:
        inputs = jnp.ones((k+2, 224, 224, 3), jnp.float32)

    #     random.normal(
    #     key_i, shape=(batch_size * (self.data_config.k + 2), H, W, C)
    # )

    # Step 2: Get backbone and its pretrained parameters
    backbone, backbone_params = _get_backbone_and_params(model_arch)

    # Step 3: Initialize the Model with backbone, num_classes, and k
    print(f'Odd K: {k}')
    model = Model(backbone=backbone, num_classes=num_classes, k=k)

    # Step 4: Initialize the Model's variables
    key = jax.random.PRNGKey(head_init_key)
    variables = model.init(key, inputs, train=True)  # todo: may need to account for OKO batch here like in trainer.py

    # Step 5: Merge pretrained backbone parameters into the model's variables
    # Convert FrozenDict to mutable dict using flax.core.unfreeze
    variables_mutable = frozen_dict.unfreeze(variables)  # Returns a regular dict

    # Update the 'backbone' parameters with pretrained backbone_params
    variables_mutable['params']['backbone'] = backbone_params['params']

    # If there are batch statistics in backbone_params, update them as well
    if 'batch_stats' in backbone_params:
        variables_mutable['batch_stats']['backbone'] = backbone_params['batch_stats']

    # Refreeze the variables back to FrozenDict
    variables = frozen_dict.freeze(variables_mutable)

    return model, variables
