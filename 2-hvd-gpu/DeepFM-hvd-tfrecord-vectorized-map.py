# ***************************************************************************************
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.                    *
#                                                                                       *
# Permission is hereby granted, free of charge, to any person obtaining a copy of this  *
# software and associated documentation files (the "Software"), to deal in the Software *
# without restriction, including without limitation the rights to use, copy, modify,    *
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to    *
# permit persons to whom the Software is furnished to do so.                            *
#                                                                                       *
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,   *
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A         *
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT    *
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION     *
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE        *
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                                *
# ***************************************************************************************

#!/usr/bin/env python
# coding=utf-8

import shutil
import os
import json
import glob
from datetime import date, timedelta
from time import time
import random
import tensorflow as tf
import horovod.tensorflow as hvd

# To use SagemakerPipe mode, we need to import below packages
from sagemaker_tensorflow import PipeModeDataset
from tensorflow.contrib.data import map_and_batch

#################### CMD Arguments ####################
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("feature_size", 0, "Number of features")
tf.app.flags.DEFINE_integer("field_size", 0, "Number of fields")
tf.app.flags.DEFINE_integer("embedding_size", 32, "Embedding size")
tf.app.flags.DEFINE_integer("num_epochs", 10, "Number of epochs")
tf.app.flags.DEFINE_integer("batch_size", 64, "Number of batch size")
tf.app.flags.DEFINE_integer("log_steps", 1000, "save summary every steps")
tf.app.flags.DEFINE_float("learning_rate", 0.0005, "learning rate")
tf.app.flags.DEFINE_float("l2_reg", 0.0001, "L2 regularization")
tf.app.flags.DEFINE_string("loss_type", "log_loss", "loss type {square_loss, log_loss}")
tf.app.flags.DEFINE_string(
    "optimizer", "Adam", "optimizer type {Adam, Adagrad, GD, Momentum}"
)
tf.app.flags.DEFINE_string("deep_layers", "256,128,64", "deep layers")
tf.app.flags.DEFINE_string("dropout", "0.5,0.5,0.5", "dropout rate")
tf.app.flags.DEFINE_boolean(
    "batch_norm", False, "perform batch normaization (True or False)"
)
tf.app.flags.DEFINE_float(
    "batch_norm_decay", 0.9, "decay for the moving average(recommend trying decay=0.9)"
)
tf.app.flags.DEFINE_string("training_data_dir", "", "training data dir")
tf.app.flags.DEFINE_string("val_data_dir", "", "validation data dir")
tf.app.flags.DEFINE_string("checkpoint_dir", "", "local checkpoint dir")
tf.app.flags.DEFINE_string(
    "servable_model_dir", "", "export servable model for TensorFlow Serving"
)
tf.app.flags.DEFINE_string(
    "task_type", "train", "task type {train, infer, eval, export}"
)
tf.app.flags.DEFINE_boolean(
    "clear_existing_model", False, "clear existing model or not"
)
tf.app.flags.DEFINE_list(
    "hosts",
    json.loads(os.environ.get("SM_HOSTS")),
    "get the all cluster instances name for distribute training",
)
tf.app.flags.DEFINE_string(
    "current_host",
    os.environ.get("SM_CURRENT_HOST"),
    "get current execute the program host name",
)
tf.app.flags.DEFINE_integer("pipe_mode", 0, "sagemaker data input pipe mode")
tf.app.flags.DEFINE_integer(
    "worker_per_host", 1, "worker process per training instance"
)
tf.app.flags.DEFINE_string(
    "training_channel_name", "", "training channel name for input_fn"
)
tf.app.flags.DEFINE_string(
    "evaluation_channel_name", "", "evaluation channel name for input_fn"
)
tf.app.flags.DEFINE_boolean(
    "enable_s3_shard",
    False,
    "whether enable S3 shard(True or False), this impact whether do dataset shard in input_fn",
)
tf.app.flags.DEFINE_boolean(
    "enable_data_multi_path",
    False,
    "whether use different dataset path for each channel, this impact how to do dataset shard(ONLY apply for Pipe mode) in input_fn",
)

# end of liangaws

# 1 1:0.5 2:0.03519 3:1 4:0.02567 7:0.03708 8:0.01705 9:0.06296 10:0.18185 11:0.02497 12:1 14:0.02565 15:0.03267 17:0.0247 18:0.03158 20:1 22:1 23:0.13169 24:0.02933 27:0.18159 31:0.0177 34:0.02888 38:1 51:1 63:1 132:1 164:1 236:1
# To use sagemaker pipe mode, we add an parameter - channel in input_fn()
def input_fn(
    filenames="", channel="training", batch_size=32, num_epochs=1, perform_shuffle=False
):
    def decode_tfrecord(batch_examples):
        # The feature definition here should BE consistent with LibSVM TO TFRecord process.
        features = tf.parse_example(
            batch_examples,
            features={
                "label": tf.FixedLenFeature([], tf.float32),
                "ids": tf.FixedLenFeature(dtype=tf.int64, shape=[FLAGS.field_size]),
                "values": tf.FixedLenFeature(
                    dtype=tf.float32, shape=[FLAGS.field_size]
                ),
            },
        )

        batch_label = features["label"]
        batch_ids = features["ids"]
        batch_values = features["values"]

        return {"feat_ids": batch_ids, "feat_vals": batch_values}, batch_label

    # Extract lines from input files using the Dataset API, can pass one filename or filename list
    if FLAGS.pipe_mode == 0:
        dataset = tf.data.TFRecordDataset(filenames)

        if FLAGS.enable_s3_shard:  # ShardedByS3Key
            dataset = dataset.shard(FLAGS.worker_per_host, hvd.local_rank())
        else:  # S3FullReplicate
            dataset = dataset.shard(hvd.size(), hvd.rank())
    else:
        print("-------enter into pipe mode branch!------------")
        dataset = PipeModeDataset(channel, record_format="TFRecord")

        number_host = len(FLAGS.hosts)

        if FLAGS.enable_data_multi_path:
            if FLAGS.enable_s3_shard == False:
                if number_host > 1:
                    index = hvd.rank() // FLAGS.worker_per_host
                    dataset = dataset.shard(number_host, index)
        else:
            if FLAGS.enable_s3_shard:
                dataset = dataset.shard(FLAGS.worker_per_host, hvd.local_rank())
            else:
                dataset = dataset.shard(hvd.size(), hvd.rank())

    dataset = dataset.batch(batch_size, drop_remainder=True)  # Batch size to use
    dataset = dataset.map(
        decode_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    if num_epochs > 1:
        dataset = dataset.repeat(num_epochs)

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def model_fn(features, labels, mode, params):
    """Bulid Model function f(x) for Estimator."""
    # ------hyperparameters----
    field_size = params["field_size"]
    feature_size = params["feature_size"]
    embedding_size = params["embedding_size"]
    l2_reg = params["l2_reg"]
    learning_rate = params["learning_rate"] * hvd.size()

    layers = list(map(int, params["deep_layers"].split(",")))
    dropout = list(map(float, params["dropout"].split(",")))

    # ------bulid weights------
    FM_B = tf.get_variable(
        name="fm_bias", shape=[1], initializer=tf.constant_initializer(0.0)
    )
    FM_W = tf.get_variable(
        name="fm_w", shape=[feature_size], initializer=tf.glorot_normal_initializer()
    )
    FM_V = tf.get_variable(
        name="fm_v",
        shape=[feature_size, embedding_size],
        initializer=tf.glorot_normal_initializer(),
    )

    # ------build feaure-------
    feat_ids = features["feat_ids"]
    feat_ids = tf.reshape(feat_ids, shape=[-1, field_size])
    feat_vals = features["feat_vals"]
    feat_vals = tf.reshape(feat_vals, shape=[-1, field_size])

    # ------build f(x)------
    with tf.variable_scope("First-order"):
        feat_wgts = tf.nn.embedding_lookup(FM_W, feat_ids)  # None * F * 1
        y_w = tf.reduce_sum(tf.multiply(feat_wgts, feat_vals), 1)

    with tf.variable_scope("Second-order"):
        embeddings = tf.nn.embedding_lookup(FM_V, feat_ids)  # None * F * K
        feat_vals = tf.reshape(feat_vals, shape=[-1, field_size, 1])
        embeddings = tf.multiply(embeddings, feat_vals)  # vij*xi
        sum_square = tf.square(tf.reduce_sum(embeddings, 1))
        square_sum = tf.reduce_sum(tf.square(embeddings), 1)
        y_v = 0.5 * tf.reduce_sum(tf.subtract(sum_square, square_sum), 1)  # None * 1

    with tf.variable_scope("Deep-part"):
        if FLAGS.batch_norm:

            if mode == tf.estimator.ModeKeys.TRAIN:
                train_phase = True
            else:
                train_phase = False
        else:
            normalizer_fn = None
            normalizer_params = None

        deep_inputs = tf.reshape(
            embeddings, shape=[-1, field_size * embedding_size]
        )  # None * (F*K)

        for i in range(len(layers)):
            deep_inputs = tf.contrib.layers.fully_connected(
                inputs=deep_inputs,
                num_outputs=layers[i],
                weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                scope="mlp%d" % i,
            )
            if FLAGS.batch_norm:
                deep_inputs = batch_norm_layer(
                    deep_inputs, train_phase=train_phase, scope_bn="bn_%d" % i
                )
            if mode == tf.estimator.ModeKeys.TRAIN:
                deep_inputs = tf.nn.dropout(deep_inputs, keep_prob=dropout[i])

        y_deep = tf.contrib.layers.fully_connected(
            inputs=deep_inputs,
            num_outputs=1,
            activation_fn=tf.identity,
            weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
            scope="deep_out",
        )
        y_d = tf.reshape(y_deep, shape=[-1])

    with tf.variable_scope("DeepFM-out"):
        y_bias = FM_B * tf.ones_like(y_d, dtype=tf.float32)  # None * 1
        y = y_bias + y_w + y_v + y_d
        pred = tf.sigmoid(y)

    predictions = {"prob": pred}
    export_outputs = {
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
            predictions
        )
    }
    # Provide an estimator spec for `ModeKeys.PREDICT`
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode, predictions=predictions, export_outputs=export_outputs
        )

    # ------bulid loss------
    loss = (
        tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=labels))
        + l2_reg * tf.nn.l2_loss(FM_W)
        + l2_reg * tf.nn.l2_loss(FM_V)
    )

    # Provide an estimator spec for `ModeKeys.EVAL`
    eval_metric_ops = {"auc": tf.metrics.auc(labels, pred)}
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            eval_metric_ops=eval_metric_ops,
        )

    # ------bulid optimizer------
    if FLAGS.optimizer == "Adam":
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8
        )
    elif FLAGS.optimizer == "Adagrad":
        optimizer = tf.train.AdagradOptimizer(
            learning_rate=learning_rate, initial_accumulator_value=1e-8
        )
    elif FLAGS.optimizer == "Momentum":
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate, momentum=0.95
        )
    elif FLAGS.optimizer == "ftrl":
        optimizer = tf.train.FtrlOptimizer(learning_rate)

    optimizer = hvd.DistributedOptimizer(optimizer)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    # Provide an estimator spec for `ModeKeys.TRAIN` modes
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
            mode=mode, predictions=predictions, loss=loss, train_op=train_op
        )


def batch_norm_layer(x, train_phase, scope_bn):
    bn_train = tf.contrib.layers.batch_norm(
        x,
        decay=FLAGS.batch_norm_decay,
        center=True,
        scale=True,
        updates_collections=None,
        is_training=True,
        reuse=None,
        scope=scope_bn,
    )
    bn_infer = tf.contrib.layers.batch_norm(
        x,
        decay=FLAGS.batch_norm_decay,
        center=True,
        scale=True,
        updates_collections=None,
        is_training=False,
        reuse=True,
        scope=scope_bn,
    )
    z = tf.cond(tf.cast(train_phase, tf.bool), lambda: bn_train, lambda: bn_infer)
    return z


def main(_):

    hvd.init()
    print("task_type ", FLAGS.task_type)
    print("checkpoint_dir ", FLAGS.checkpoint_dir)
    print("training_data_dir ", FLAGS.training_data_dir)
    print("val_data_dir ", FLAGS.val_data_dir)
    print("num_epochs ", FLAGS.num_epochs)
    print("feature_size ", FLAGS.feature_size)
    print("field_size ", FLAGS.field_size)
    print("embedding_size ", FLAGS.embedding_size)
    print("batch_size ", FLAGS.batch_size)
    print("deep_layers ", FLAGS.deep_layers)
    print("dropout ", FLAGS.dropout)
    print("loss_type ", FLAGS.loss_type)
    print("optimizer ", FLAGS.optimizer)
    print("learning_rate ", FLAGS.learning_rate)
    print("batch_norm_decay ", FLAGS.batch_norm_decay)
    print("batch_norm ", FLAGS.batch_norm)
    print("l2_reg ", FLAGS.l2_reg)

    if FLAGS.pipe_mode == 0:
        tr_files = glob.glob(
            r"%s/**/tr*.tfrecords" % FLAGS.training_data_dir, recursive=True
        )
        random.shuffle(tr_files)
        va_files = glob.glob(
            r"%s/**/va*.tfrecords" % FLAGS.val_data_dir, recursive=True
        )
        te_files = glob.glob(
            r"%s/**/te*.tfrecords" % FLAGS.val_data_dir, recursive=True
        )
    else:
        tr_files = ""
        va_files = ""
        te_files = ""

    print("tr_files:", tr_files)
    print("va_files:", va_files)
    print("te_files:", te_files)

    if FLAGS.clear_existing_model:
        try:
            shutil.rmtree(FLAGS.checkpoint_dir)
        except Exception as e:
            print(e, "at clear_existing_model")
        else:
            print("existing model cleaned at %s" % FLAGS.checkpoint_dir)

    # ------bulid Tasks------
    model_params = {
        "field_size": FLAGS.field_size,
        "feature_size": FLAGS.feature_size,
        "embedding_size": FLAGS.embedding_size,
        "learning_rate": FLAGS.learning_rate,
        "batch_norm_decay": FLAGS.batch_norm_decay,
        "l2_reg": FLAGS.l2_reg,
        "deep_layers": FLAGS.deep_layers,
        "dropout": FLAGS.dropout,
    }

    # to use Horovod, pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    # save checkpoints only on worker 0 to prevent other workers from corrupting them.
    print("current horovod rank is ", hvd.rank())
    print("host is ", FLAGS.hosts)
    print("current host is ", FLAGS.current_host)

    if hvd.rank() == 0:
        DeepFM = tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir=FLAGS.checkpoint_dir,
            params=model_params,
            config=tf.estimator.RunConfig().replace(session_config=config),
        )
    else:
        DeepFM = tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir=None,
            params=model_params,
            config=tf.estimator.RunConfig().replace(session_config=config),
        )

    # BroadcastGlobalVariablesHook broadcasts initial variable states from rank 0 to all other processes. This is necessary to ensure consistent initialization of all workers when training is started with random weights or restored from a checkpoint.
    bcast_hook = hvd.BroadcastGlobalVariablesHook(0)

    channel_names = json.loads(os.environ["SM_CHANNELS"])
    print("channel name", channel_names)
    print("first channel", channel_names[0])
    print("last channel name", channel_names[-1])
    eval_channel = channel_names[0]

    if FLAGS.task_type == "train":

        if FLAGS.pipe_mode == 0:  # file mode
            for _ in range(FLAGS.num_epochs):
                DeepFM.train(
                    input_fn=lambda: input_fn(
                        tr_files, num_epochs=1, batch_size=FLAGS.batch_size
                    ),
                    hooks=[bcast_hook],
                )
                if hvd.rank() == 0:
                    DeepFM.evaluate(
                        input_fn=lambda: input_fn(
                            va_files, num_epochs=1, batch_size=FLAGS.batch_size
                        )
                    )
        else:
            DeepFM.train(
                input_fn=lambda: input_fn(
                    channel=channel_names[1 + hvd.local_rank()],
                    num_epochs=FLAGS.num_epochs,
                    batch_size=FLAGS.batch_size,
                ),
                hooks=[bcast_hook],
            )
            if hvd.rank() == 0:
                DeepFM.evaluate(
                    input_fn=lambda: input_fn(
                        channel=eval_channel, num_epochs=1, batch_size=FLAGS.batch_size
                    )
                )

    elif FLAGS.task_type == "eval":
        DeepFM.evaluate(
            input_fn=lambda: input_fn(
                va_files, num_epochs=1, batch_size=FLAGS.batch_size
            )
        )
    elif FLAGS.task_type == "infer":
        preds = DeepFM.predict(
            input_fn=lambda: input_fn(
                te_files, num_epochs=1, batch_size=FLAGS.batch_size
            ),
            predict_keys="prob",
        )
        with open(FLAGS.val_data_dir + "/pred.txt", "w") as fo:
            for prob in preds:
                fo.write("%f\n" % (prob["prob"]))

    if FLAGS.task_type == "export" or FLAGS.task_type == "train":

        feature_spec = {
            "feat_ids": tf.placeholder(
                dtype=tf.int64, shape=[None, FLAGS.field_size], name="feat_ids"
            ),
            "feat_vals": tf.placeholder(
                dtype=tf.float32, shape=[None, FLAGS.field_size], name="feat_vals"
            ),
        }
        serving_input_receiver_fn = (
            tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
        )

        # Save model and history only on worker 0 (i.e. master)
        if hvd.rank() == 0:
            DeepFM.export_savedmodel(
                FLAGS.servable_model_dir, serving_input_receiver_fn
            )


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
