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
import os
import numpy as np
import tensorflow as tf


def convert_tfrecords(input_filename, output_filename):
    """Concert the LibSVM contents to TFRecord.
    Args:
    input_filename: LibSVM filename.
    output_filename: Desired TFRecord filename.
    """
    print("Starting to convert {} to {}...".format(input_filename, output_filename))
    writer = tf.python_io.TFRecordWriter(output_filename)

    try:
        for line in open(input_filename, "r"):
            data = line.split(" ")
            label = float(data[0])
            ids = []
            values = []
            for fea in data[1:]:
                id, value = fea.split(":")
                ids.append(int(id))
                values.append(float(value))
            # Write samples one by one
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "label": tf.train.Feature(
                            float_list=tf.train.FloatList(value=[label])
                        ),
                        "ids": tf.train.Feature(
                            int64_list=tf.train.Int64List(value=ids)
                        ),
                        "values": tf.train.Feature(
                            float_list=tf.train.FloatList(value=values)
                        ),
                    }
                )
            )
            writer.write(example.SerializeToString())
    finally:
        writer.close()

    print("Successfully converted {} to {}!".format(input_filename, output_filename))


sess = tf.InteractiveSession()

try:
    convert_tfrecords(
        "/home/ec2-user/SageMaker/deepfm test/tfrecord_file/tr.libsvm",
        "/home/ec2-user/SageMaker/deepfm test/tfrecord_file/tfrecord/train.tfrecords",
    )
    convert_tfrecords(
        "/home/ec2-user/SageMaker/deepfm test/tfrecord_file/va.libsvm",
        "//home/ec2-user/SageMaker/deepfm test/tfrecord_file/tfrecord/va.tfrecords",
    )
finally:
    sess.close()
