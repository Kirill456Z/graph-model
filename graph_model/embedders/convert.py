# tensorflow==1.14.0
# tensorflow_text==0.1.0
# tensorflow-hub==0.7.0
# wget http://files.deeppavlov.ai/alexaprize_data/convert_reddit_v2.8.tar.gz
# tar xzfv .....
# MODEL_PATH=........../convert_data/convert
import tensorflow_hub as tfhub
import tensorflow_text
import tensorflow as tf

import numpy as np

from tqdm import tqdm

# MODEL_PATH = os.getenv("MODEL_PATH")
MODEL_PATH = "/home/akhmadjonov/workspace/Convert_TF2"
# MODEL_PATH = "https://github.com/davidalami/ConveRT/releases/download/1.0/multicontext_tf_model.tar"

# enable GPU growth
tf.config.experimental.enable_tensor_float_32_execution(False)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

# The following setting allows the Tf1 model to run in Tf2
tf.compat.v1.disable_eager_execution()

#setting the logging verbosity level to errors-only
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

sess = tf.compat.v1.Session()
text_placeholder = tf.compat.v1.placeholder(dtype=tf.string, shape=[None])

module = tfhub.Module(MODEL_PATH)

extra_text_placeholder = tf.compat.v1.placeholder(dtype=tf.string, shape=[None])

# The encode_context signature now also takes the extra context.
context_encoding_tensor = module(
    {
        "context": text_placeholder, 
        "extra_context": extra_text_placeholder
    },
    signature="encode_context"
)

responce_text_placeholder = tf.compat.v1.placeholder(dtype=tf.string, shape=[None])

response_encoding_tensor = module(responce_text_placeholder,
                                  signature="encode_response")

sess.run(tf.compat.v1.tables_initializer())
sess.run(tf.compat.v1.global_variables_initializer())


def encode_context(dialogue_history):
    """Encode the dialogue context to the response ranking vector space.

    Args:
        dialogue_history: a list of strings, the dialogue history, in
            chronological order.
    """

    # The context is the most recent message in the history.
    context = dialogue_history[-1]

    extra_context = list(dialogue_history[:-1])
    extra_context.reverse()
    extra_context_feature = " ".join(extra_context)

    return sess.run(
        context_encoding_tensor,
        feed_dict={
            text_placeholder: [context],
            extra_text_placeholder: [extra_context_feature]
        },
    )[0]


def encode_dialogues(dialogues):
    histories = [" ".join(u.utterance for u in dialogue[:k][::-1])
                 for dialogue in dialogues
                 for k in range(len(dialogue))]
    utterances = dialogues.utterances

    print(len(histories))
    print(len(utterances))

    parts = []
    step = 1000
    for i in tqdm(0, len(utterances), step):
        parts.append(sess.run(
            context_encoding_tensor,
            feed_dict={text_placeholder: utterances[i:i + step],
                       extra_text_placeholder: histories[i:i + step]},
        )[0])
    return parts


def encode_responses(texts):
    return sess.run(response_encoding_tensor, feed_dict={responce_text_placeholder: texts})

def l2_normalize(encodings):
    """L2 normalizes the given matrix of encodings."""
    norms = np.linalg.norm(encodings, ord=2, axis=-1, keepdims=True)
    return encodings / norms