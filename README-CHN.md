# deepfm-tensorflow-distributed-training-on-sagemaker

在该示例中，我们将给出两个基于Amazon SageMaker的 deepfm 分布式训练样例, 一个基于TensorFlow Parameter Server + CPU，一个基于Horovod + GPU，而这两个也是我们见到的用的比较多的方式。

[English README](https://github.com/aws-samples/deepfm-tensorflow-distributed-training-on-sagemaker/blob/main/README-EN.md)




## 代码几点说明：
- 本示例支持的 TensorFlow 版本为1.14， 1.15.2

- 训练数据格式为 TFRecord，关于如何从 libsvm转为 tfrecord可以参考 **tools** 目录下的脚本。本示例中我们提供了样例数据，并放在了data目录下。

### Parameter Server CPU

- 在SageMaker TF Parameter Server（简称PS）模式下，每个训练实例上会有一个进程用于parameter server，每个实例上有一个worker，并且应用的PS异步模式的；

- SageMaker会设置 PS所需要的环境变量，如Master，worker，task，chief，job等信息，无需在代码中额外设置；

- 设置MKL-DNN CPU bind 策略和线程池，以提高CPU使用率

  ```python
  os.environ["KMP_AFFINITY"]= "verbose,disabled"
  #os.environ["KMP_AFFINITY"]= "granularity=fine,compact,1,0"
  #os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,scatter,1,0"
  os.environ['OMP_NUM_THREADS'] = str(num_cpus)
  os.environ['KMP_SETTINGS'] = '1'
  ```

- 设置intra_op_parallelism_threads为当前VCPU数量来并行单个计算图操作的并行度，设置inter_op_parallelism_threads为没有依赖的多个操作的并行度，多个操作的每个操作如果本身也能并行的话，会共享intra_op_parallelism_threads设置的线程池。若inter_op_parallelism_threads设置为0则让tensorflow自己来选择合适的多个操作的并行度。

  ```python
  config = tf.ConfigProto(allow_soft_placement=True, device_count={'CPU': num_cpus}, intra_op_parallelism_threads=num_cpus, inter_op_parallelism_threads=num_cpus, device_filters=device_filters)
  ```

- Python脚本中，我们有如下一段代码：

  该代码的用途是在多个主机的时候，需要在每个主机上面设置**device_filters**，并用于 tf.ConfigProto() 中，否则的话会出现模型训练完成时，master worker做完eval之后会一直hang住，而non master worker这个时候是训练完毕的状态。

  修改完了之后，在训练的时候一定要调用 **tf.estimator.train_and_evaluate(self.model, train_spec, eval_spec)** API，如果调用**tf.estimator.train** API的话，会在开始训练的时候就一直卡住（这也是 TF 的 known issue）。

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

- model_dir 是一个S3路径（每次启动训练时用不同的路径以使模型从0开始训练，在helper code中我们在S3路径上加了一个时间戳的后缀），用于 TF 保存checkpoint，TF在PS模式下，需要一个共享存储来保存ckpt。通常训练开始时，master worker初始化模型参数，然后传给PS，其他worker从PS中获取模型参数（通常晚于master worker 5s开始），开始训练。

### SageMaker inputs S3 Shard功能

SageMaker中提供了S3侧对数据Shard的功能，使用时，将会基于文件名前缀做Shard（基于主机数量），这个Shard是按文件级别的，因此是文件数量需要被主机数量整除，同时每个文件中样本数量需要一致。这会带来数据准备阶段的开销，如果不在此处启用Shard，则数据集会被FULL下载到训练实例中，就需要在代码中根据情况做Shard（如 dataset.shard(数据分多少份，当前worker拿第几份)）。当单个主机上有多个worker的时候，即使在S3侧做了Shard，还需要在代码中进一步Shard，S3侧Shard只是按主机数量进行Shard，而不是worker数量。

```python
from sagemaker.inputs import TrainingInput

train_input = TrainingInput(train_s3_uri, distribution='ShardedByS3Key')
inputs = {'training' : train_input}
estimator.fit(inputs)
```

### Horovod GPU

- 为了在Sagemaker pipe mode下使用horovod的单机多个worker进程，需要在调用Sagemaker的estimator fit的时候用多个channel，至少单机的每个worker需要一个channel。在该示例中，我们使用 ml.p3.8xlarge实例，该实例有4块V100的卡，在使用PIPE模式时，需要注意在helper code中需要创建4个channel。

- 从SM设置的环境变量SM_CHANNELS可以获得当前的所有channel名字，之后每个worker用单独的channel来进行数据读取。这里channel名字的顺序与调用Sagemaker estimator fit时候写入的顺序是不同的。比如对于**{'training':train_s3, 'training-2':train2_s3, 'evaluation': validate_s3}**这样的三个channel，环境变量被SM设置为**['evaluation', 'training', 'training-2']**，也就是说最后一个channel 'evaluation'出现在环境变量SM_CHANNELS中的第一个，其他channel则是按照原来顺序排列。

- 关于Channel，一个主机上面每个worker一个channel（至少一个），同一个channel不能被同一个主机上不同worker使用，但是可以被其他主机上的worker使用。

- 每个channel可以对应同一个S3路径或者不同的S3路径（channel多路径模式），当对应不同S3路径时，相当于上传到S3时已经做过一次数据的Shard，而配合SageMaker inputs S3 Shard功能，训练代码中甚至可以不再需要做Shard。因此是否使用channel多路径模式以及是否使用SageMaker inputs S3 Shard功能，两两组合，在训练代码读取数据做Shard时代码各不一样。

  

  | 是否使用channel多路径 | 是否使用SageMaker inputs S3 Shard |                 训练脚本中数据 Shard部分代码                 |
  | :-------------------: | :-------------------------------: | :----------------------------------------------------------: |
  |           Y           |                 Y                 |                        无需再做Shard                         |
  |           Y           |                 N                 | dataset.shard(主机数量, 当前主机在全部主机中的index)即`dataset.shard(num_hosts, host_index)` |
  |           N           |                 Y                 | dataset.shard(每个主机worker数, 当前worker在当前主机上的index)即`dataset.shard(woker_per_host, hvd.local_rank())` |
  |           N           |                 N                 | dataset.shard(worker总数, 当前worker在worker总数中的index)即`dataset.shard(hvd.size(), hvd.rank())` |

对应的代码为：

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