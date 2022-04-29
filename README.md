# deepfm-tensorflow-distributed-training-on-amazon-sagemaker

In this demo, we show two samples about deepfm distributed training on Amazon SageMaker, one is based on Tensorflow Parameter Server on CPU and the other one is based on Horovod on GPU.

[中文 README](https://github.com/aws-samples/deepfm-tensorflow-distributed-training-on-amazon-sagemaker/blob/main/README-CHN.md)


## Tips
- TensorFlow versions 1.14 and 1.15.2 are supported in this example.

- The training dataset format is TFRecord, you could refer to the script under **tools** to convert libsvm to tfrecord. In this example, we offer sample dataset under **data** folder now for testing.

### Parameter Server CPU

- In SageMaker TF Parameter Server (PS for short) mode，each instance will have a parameter server process, each instance has one worker, the PS is async mode.

- SageMaker will set all the environment variables that PS need, such as Master, worker, task, chief and job, etc. You don't need set them in your training code.

- Set MKL-DNN CPU bind policy and thread pool to improve CPU usage.

  ```python
  os.environ["KMP_AFFINITY"]= "verbose,disabled"
  #os.environ["KMP_AFFINITY"]= "granularity=fine,compact,1,0"
  #os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,scatter,1,0"
  os.environ['OMP_NUM_THREADS'] = str(num_cpus)
  os.environ['KMP_SETTINGS'] = '1'
  ```

- Set **intra_op_parallelism_threads** to the current number of VCPUs to parallelize the parallelism of a single calculation graph operation, and set **inter_op_parallelism_threads** to the parallelism of multiple operations without dependencies. If each operation of multiple operations can be parallelized, it will share the thread pool set by intra_op_parallelism_threads. . If **inter_op_parallelism_threads** is set to 0, tensorflow will choose the appropriate parallelism of multiple operations.

  ```python
  config = tf.ConfigProto(allow_soft_placement=True, device_count={'CPU': num_cpus}, intra_op_parallelism_threads=num_cpus, inter_op_parallelism_threads=num_cpus, device_filters=device_filters)
  ```

- In the training script, we have below code:

  When use PS in multi instances, you need to set **device_filters** on each instance, and apply to tf.ConfigProto(), otherwise the master worker will hang when it finish the model traning and evaluation while non master work is in job done status.

  After set device_filters in config, you need use **tf.estimator.train_and_evaluate(self.model, train_spec, eval_spec)** instead of **tf.estimator.train** , otherwise it would hang at the begining of training. (This is a TF known issue)

```python
if len(FLAGS.hosts) > 1:
    tf_config = json.loads(os.environ['TF_CONFIG'])
    print("tf_config is ", tf_config)
    index = tf_config["task"]["index"]
    print("index is ", index)

    #每个训练实例都会有一个parameter server进程，所以每个实例都需要一个ps的device filter
    device_filters = ['/job:ps']
    if str(tf_config["task"]["type"]) == 'master':
        device_filters.append('/job:master')
    else:
        worker_index = '/job:worker/task:' + str(index)
        device_filters.append(worker_index)

    config = tf.ConfigProto(allow_soft_placement=True, device_count={'CPU': num_cpus}, intra_op_parallelism_threads=num_cpus, inter_op_parallelism_threads=num_cpus, device_filters=device_filters)
else:
    config = tf.ConfigProto(allow_soft_placement=True, device_count={'CPU': num_cpus}, intra_op_parallelism_threads=num_cpus, inter_op_parallelism_threads=num_cpus)

run_config = tf.estimator.RunConfig().replace(session_config = config)
```

- model_dir is an S3 path (each time you start training, use a different path to make the model start training from 0. In the helper code, we add a timestamp suffix to the S3 path), used for TF to save checkpoint. When use TF in PS mode, a shared storage is needed to save ckpt. Usually at the beginning of training, the master worker initializes the model parameters and then passes them to the PS, and the other workers obtain the model parameters from the PS (usually starting later than the master worker 5 seconds) and start training.

### SageMaker inputs S3 Shard

SageMaker provides the data shard feature on the S3 side. When using it, the shard will be based on the file name prefix (based on the number of hosts). This shard is based on the file level, so the number of files needs to be evenly divided by the number of hosts. The number of samples in each file needs to be consistent. This will bring the overhead of the data preparation stage. If the Shard is not enabled here, the dataset will be FULL downloaded to the training instances. You need to make the Shard in the code according to the situation (such as **dataset.shard (the num of data need to be divided, the index of dataset that the current worker to take)**). When there are multiple workers on a single host, even if shards are made on the S3 side, further shards are required in the code. Shards on the S3 side only shards according to the number of hosts, not the number of workers.

```python
from sagemaker.inputs import TrainingInput

train_input = TrainingInput(train_s3_uri, distribution='ShardedByS3Key')
inputs = {'training' : train_input}
estimator.fit(inputs)
```

### Horovod GPU

- In order to use horovod on single host multiple worker with Sagemaker pipe mode, you need to use multiple channels when calling Sagemaker estimator fit, each worker on a single host need at least one channel. In this example, we use the ml.p3.8xlarge instance for example, it has 4 V100 GPU cards. When using the PIPE mode, we need to set 4 channels in the helper code.

- From the environment variable **SM_CHANNELS** set by SageMaker, you can get the names of all channels, and then each worker uses a separate channel to read data. The order of channel names here is different from the order of when calling Sagemaker estimator fit. For example, for three channels like **{'training':train_s3,'training-2':train2_s3,'evaluation': validate_s3}**, the **SM_CHANNELS** environment variable are set to **['evaluation','training', 'training-2']**, which means that the last channel **'evaluation'** appears first in the environment variable **SM_CHANNELS**, and the other channels are arranged in the original order. 

- Regarding Channel, each worker on a host has a channel (at least one). The same channel cannot be used by different workers on the same host, but can be used by other workers on other hosts.

- Each channel can correspond to the same S3 path or different S3 paths (channel multi-path mode). When corresponding to different S3 paths, it is equivalent to a shard that has already done when uploading dataset to S3, and cooperates with SageMaker inputs S3 Shard , You don’t even need to do shards in the training code. Therefore, whether to use the channel multi-path mode and whether to use the SageMaker inputs S3 Shard feature, the two-by-two combination, the data load code is different accroding to how to do data shard.

  | Use channel multi-path? | Use SageMaker inputs S3 Shard? |          The dataset shard code in training script           |
  | :---------------------: | :----------------------------: | :----------------------------------------------------------: |
  |            Y            |               Y                |                       No need to Shard                       |
  |            Y            |               N                | dataset.shard(number of hosts, the index of current host in the cluster), `dataset.shard(num_hosts, host_index)` |
  |            N            |               Y                | dataset.shard(number of workers on each instance, the index of current worker in the current host), `dataset.shard(woker_per_host, hvd.local_rank())` |
  |            N            |               N                | dataset.shard(the number of total workers,  the index of current worker in all the workers), `dataset.shard(hvd.size(), hvd.rank())` |

The related code is:

```python
dataset = PipeModeDataset(channel, record_format='TFRecord')
number_host = len(FLAGS.hosts)
if FLAGS.enable_data_multi_path : 
    if FLAGS.enable_s3_shard == False :
        if number_host > 1:
            index = hvd.rank() // FLAGS.worker_per_host
            dataset = dataset.shard(number_host, index)
else :
    if FLAGS.enable_s3_shard :
        dataset = dataset.shard(FLAGS.worker_per_host, hvd.local_rank())
else :
    dataset = dataset.shard(hvd.size(), hvd.rank())
```
