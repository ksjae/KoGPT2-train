import sys
import os
import argparse
import json
import re

import tensorflow.compat.v1 as tf
import numpy as np

from train.modeling import GroverModel, GroverConfig, sample
from tokenization import tokenization

import argparse
import logging
from tqdm import trange

import socket

##### ignore tf deprecated warning temporarily
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.DEBUG)
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
try:
    from tensorflow.python.util import module_wrapper as deprecation
except ImportError:
    from tensorflow.python.util import deprecation_wrapper as deprecation
deprecation._PER_MODULE_WARNING_LIMIT = 0
#####

parser = argparse.ArgumentParser(description='Contextual generation (aka given some metadata we will generate articles')
parser.add_argument(
    '-metadata_fn',
    dest='metadata_fn',
    type=str,
    help='Path to a JSONL containing metadata',
)
parser.add_argument(
    '-out_fn',
    dest='out_fn',
    type=str,
    help='Out jsonl, which will contain the completed jsons',
)
parser.add_argument(
    '-input',
    dest='input',
    type=str,
    help='Text to complete',
)
parser.add_argument(
    '-config_fn',
    dest='config_fn',
    default='configs/large.json',
    type=str,
    help='Configuration JSON for the model',
)
parser.add_argument(
    '-ckpt_fn',
    dest='ckpt_fn',
    default='models/large/model.ckpt-75000',
    type=str,
    help='checkpoint file for the model',
)
parser.add_argument(
    '-target',
    dest='target',
    default='article',
    type=str,
    help='What to generate for each item in metadata_fn. can be article (body), title, etc.',
)
parser.add_argument(
    '-batch_size',
    dest='batch_size',
    default=1,
    type=int,
    help='How many things to generate per context. will split into chunks if need be',
)
parser.add_argument(
    '-num_folds',
    dest='num_folds',
    default=1,
    type=int,
    help='Number of folds. useful if we want to split up a big file into multiple jobs.',
)
parser.add_argument(
    '-fold',
    dest='fold',
    default=0,
    type=int,
    help='which fold we are on. useful if we want to split up a big file into multiple jobs.'
)
parser.add_argument(
    '-max_batch_size',
    dest='max_batch_size',
    default=None,
    type=int,
    help='max batch size. You can leave this out and we will infer one based on the number of hidden layers',
)
parser.add_argument(
    '-top_p',
    dest='top_p',
    default=0.95,
    type=float,
    help='p to use for top p sampling. if this isn\'t none, use this for everthing'
)
parser.add_argument(
    '-min_len',
    dest='min_len',
    default=128,
    type=int,
    help='min length of sample',
)
parser.add_argument(
    '-eos_token',
    dest='eos_token',
    default=50256,
    type=int,
    help='eos token id',
)
parser.add_argument(
    '-samples',
    dest='samples',
    default=1,
    type=int,
    help='num_samples',
)

def extract_generated_target(output_tokens, tokenizer):
    """
    Given some tokens that were generated, extract the target
    :param output_tokens: [num_tokens] thing that was generated
    :param encoder: how they were encoded
    :param target: the piece of metadata we wanted to generate!
    :return:
    """
    # Filter out first instance of start token
    assert output_tokens.ndim == 1

    start_ind = 0
    end_ind = output_tokens.shape[0]

    return {
        'extraction': tokenization.printable_text(tokenizer.convert_ids_to_tokens(output_tokens)),
        'start_ind': start_ind,
        'end_ind': end_ind,
    }

args = parser.parse_args()
proj_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
tokenizer = tokenization.FullTokenizer(vocab_file='./kotok' , do_lower_case=True)
news_config = GroverConfig.from_json_file(args.config_fn)

# We might have to split the batch into multiple chunks if the batch size is too large
default_mbs = {12: 32, 24: 16, 48: 3}
max_batch_size = args.max_batch_size if args.max_batch_size is not None else default_mbs[news_config.num_hidden_layers]

# factorize args.batch_size = (num_chunks * batch_size_per_chunk) s.t. batch_size_per_chunk < max_batch_size
num_chunks = int(np.ceil(args.batch_size / max_batch_size))
batch_size_per_chunk = int(np.ceil(args.batch_size / num_chunks))

# This controls the top p for each generation.
top_p = np.ones((num_chunks, batch_size_per_chunk), dtype=np.float32) * args.top_p

def gen(text, length):
    line = tokenization.convert_to_unicode(text)
    encoded = tokenizer.tokenize(line)
    context_formatted = []
    context_formatted.extend(encoded)
    # Format context end

    gens = []
    gens_raw = []
    gen_probs = []

    for chunk_i in range(num_chunks):
        tokens_out, probs_out = sess.run([tokens, probs],
                                        feed_dict={initial_context: [context_formatted] * batch_size_per_chunk,
                                                    eos_token: args.eos_token, min_len: int(length),
                                                    p_for_topp: top_p[chunk_i]})

        for t_i, p_i in zip(tokens_out, probs_out):
            extraction = extract_generated_target(output_tokens=t_i, tokenizer=tokenizer)
            gens.append(extraction['extraction'])

    l = re.findall('.{1,70}', gens[0].replace('[UNK]', '').replace('to', ''))
    result = "".join(l)
    return result


if __name__ == '__main__':
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    with tf.device('/device:XLA_GPU:0'):
        with tf.Session(config=tf_config, graph=tf.Graph()) as sess:
            initial_context = tf.placeholder(tf.int32, [batch_size_per_chunk, None])
            p_for_topp = tf.placeholder(tf.float32, [batch_size_per_chunk])
            eos_token = tf.placeholder(tf.int32, [])
            min_len = tf.placeholder(tf.int32, [])
            tokens, probs = sample(news_config=news_config, initial_context=initial_context,
                                eos_token=eos_token, min_len=min_len, ignore_ids=None, p_for_topp=p_for_topp,
                                do_topk=False)

            saver = tf.train.Saver()
            saver.restore(sess, args.ckpt_fn)
            print('🍺Model loaded. \nLet\'s serve!')
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', '47862'))
                s.listen()
                conn, addr = s.accept()
                with conn:
                    data = b""
                    while True:
                        data_pt = conn.recv(1024)
                        if not data_pt:
                            break
                        data += data_pt
                    text, length = ast.literal_eval(data)
                    conn.sendall(repr(gen(text, length)))