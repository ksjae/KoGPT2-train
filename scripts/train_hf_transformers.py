from datasets import Dataset
from datasets.utils import map_nested
from functools import partial, wraps
import numpy as np
import pandas as pd
from transformers.modeling_tf_gpt2 import TFGPT2LMHeadModel
from transformers.file_utils import hf_bucket_url, TF2_WEIGHTS_NAME
import tensorflow as tf

class FixedDataset(Dataset):
    def _convert_outputs(
        self, outputs, format_type=None, format_columns=None, output_all_columns=False, format_kwargs=None
    ):
        format_kwargs = format_kwargs if format_kwargs is not None else {}
        if format_type is None:
            if output_all_columns:
                return outputs
            if isinstance(outputs, dict) and format_columns is not None:
                return {k: v for k, v in outputs.items() if k in format_columns}
            return outputs

        map_nested_kwargs = {}
        if format_type == "numpy":
            if "copy" not in format_kwargs:
                format_kwargs["copy"] = False
            command = partial(np.array, **format_kwargs)
            map_nested_kwargs["map_list"] = False  # convert lists to arrays
        elif format_type == "torch":
            import torch

            map_nested_kwargs["map_list"] = False  # convert lists to tensors

            def command(x):
                if isinstance(
                    x, (list, tuple, np.ndarray)
                ):  # add support for nested types like struct of list of struct
                    x = np.array(x, copy=False)
                    if x.dtype == np.object:  # pytorch tensors cannot be instantied from an array of objects
                        return [map_nested(command, i, **map_nested_kwargs) for i in x]
                return torch.tensor(x, **format_kwargs)

        elif format_type == "tensorflow":
            import tensorflow

            map_nested_kwargs["map_list"] = False  # convert lists to tensors

            def command(x):
                if isinstance(
                    x, (list, tuple, np.ndarray)
                ):  # add support for nested types like struct of list of struct
                    x = np.array(x, copy=False)
                    x = x[:2048]
                    x = np.pad(x, (0,2048-len(x)), 'constant', constant_values=0)
                    if x.dtype == np.object:  # tensorflow tensors can sometimes be instantied from an array of objects
                        try:
                            return tensorflow.constant(x, **format_kwargs)
                        except ValueError:
                            return [map_nested(command, i, **map_nested_kwargs) for i in x]
                t = tensorflow.constant(x, **format_kwargs)
                #paddings = [[0,0], [0,2048-tf.shape(t)[0]]]
                #return tf.pad(t, paddings, 'CONSTANT', constant_values=-1)
                return t

        else:

            def identity(x):
                return x

            command = identity
        if isinstance(outputs, (list, tuple, np.ndarray, pd.Series)):
            return command(outputs)
        elif isinstance(outputs, pd.DataFrame):
            if format_columns is not None and not output_all_columns:
                to_remove_columns = [col for col in self.column_names if col not in format_columns]
                output_dict = outputs.drop(to_remove_columns, axis=1)
            else:
                output_dict = outputs
        else:
            output_dict = {}
            for k, v in outputs.items():
                if format_columns is not None and k not in format_columns and not output_all_columns:
                    continue
                if format_columns is None or k in format_columns:
                    v = map_nested(command, v, **map_nested_kwargs)
                output_dict[k] = v
        return output_dict

ds = FixedDataset.from_file('../WRITTEN/dataset.arrow')
ds.set_format(type='tensorflow', columns=['input_ids'], shape=[2048])
mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
with mirrored_strategy.scope():
    config_name = 'gpt2'
    model = TFGPT2LMHeadModel.from_pretrained(config_name)
    gpt2_weights_file_url = hf_bucket_url(config_name, filename=TF2_WEIGHTS_NAME)
    gpt2_weights_file = cached_path(bert_weights_file_url)
    model.load_weights(gpt2_weights_file, by_name=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss)
    model.fit(tf.data.Dataset.from_tensor_slices(ds['input_ids']), epochs=2, steps_per_epoch=115)