import math
import tempfile
import time
import tensorflow as tf
import fileUtil as fileUtil
from tensorflow.examples.tutorials.mnist import input_data

flags = tf.app.flags
IMAGE_PIXELS = 28

flags.DEFINE_string('data_dir', '/tmp/mnist-data', 'Directory  for storing mnist data')
flags.DEFINE_integer('hidden_units', 100, 'Number of units in the hidden layer of the NN')
flags.DEFINE_integer('train_steps', 10000, 'Number of training steps to perform')
flags.DEFINE_integer('batch_size', 100, 'Training batch size ')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')

# 定义分布式参数
# 参数服务器parameter server节点
ps_hosts = fileUtil.get_config('ps_hosts','ps_hosts')
flags.DEFINE_string('ps_hosts', ps_hosts,
                    'Comma-separated list of hostname:port pairs')

# 两个worker节点
worker_hosts = fileUtil.get_config('worker_hosts','worker_hosts')
flags.DEFINE_string('worker_hosts', worker_hosts,
                    'Comma-separated list of hostname:port pairs')


# 设置job name参数
flags.DEFINE_string('job_name', None, 'job name: worker or ps')

# 设置任务的索引
flags.DEFINE_integer('task_index', None, 'Index of task within the job')

# 选择异步并行，同步并行
flags.DEFINE_integer("issync", None, "是否采用分布式的同步模式，1表示同步模式，0表示异步模式")

FLAGS = flags.FLAGS

def main(unused_argv):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    if FLAGS.job_name is None or FLAGS.job_name == '':
        raise ValueError('job_name不能为空！')
    else:
        print ('job_name是 : %s' % FLAGS.job_name)
    if FLAGS.task_index is None or FLAGS.task_index == '':
        raise ValueError('task_index索引不能为空!')
    else:
        print ('task_index索引是 : %d' % FLAGS.task_index)

    #参数服务器列表
    ps_spec = FLAGS.ps_hosts.split(',')
    #work节点列表
    worker_spec = FLAGS.worker_hosts.split(',')

    # 创建集群，包含ps 和 worker
    # num_worker = len(worker_spec)
    cluster = tf.train.ClusterSpec({'ps': ps_spec, 'worker': worker_spec})
    #创建server
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    #如果角色名字是ps的话， 程序就join到这里，作为参数更新的服务
    if FLAGS.job_name == 'ps':
        print('等待终端连接...')
        server.join()

    is_chief = (FLAGS.task_index == 0) #0号worker为chief,task_index = 0的worker设置成chief supervisors
    with tf.device(tf.train.replica_device_setter(
            cluster=cluster
    )):
        # 创建纪录全局训练步数变量
        global_step = tf.Variable(0, name='global_step', trainable=False)
        hid_w = tf.Variable(tf.truncated_normal([IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units],
                                                stddev=1.0 / IMAGE_PIXELS), name='hid_w')
        hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name='hid_b')

        sm_w = tf.Variable(tf.truncated_normal([FLAGS.hidden_units, 10],
                                               stddev=1.0 / math.sqrt(FLAGS.hidden_units)), name='sm_w')
        sm_b = tf.Variable(tf.zeros([10]), name='sm_b')

        x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
        y_ = tf.placeholder(tf.float32, [None, 10])

        hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
        hid = tf.nn.relu(hid_lin)

        y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))
        cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
        opt = tf.train.AdamOptimizer(FLAGS.learning_rate)

        train_step = opt.minimize(cross_entropy, global_step=global_step)

        # 生成本地的参数初始化操作init_op
        init_op = tf.global_variables_initializer()
        train_dir = tempfile.mkdtemp()
        #创建Supervisor，管理session
        sv = tf.train.Supervisor(is_chief=is_chief, logdir=train_dir, init_op=init_op, recovery_wait_secs=1,
                                 global_step=global_step)

        if is_chief:
            print('Worker %d: 初始化 session...' % FLAGS.task_index)
        else:
            print('Worker %d: 等待session初始化...' % FLAGS.task_index)

        #启动sess
        sess = sv.prepare_or_wait_for_session(server.target)
        # with sv.prepare_or_wait_for_session(server.target) as sess:
        print('Worker %d: session初始化完毕.' % FLAGS.task_index)

        time_begin = time.time()
        print('Traing begins @ %f' % time_begin)

        local_step = 0
        while True:
            batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
            train_feed = {x: batch_xs, y_: batch_ys}

            _, step = sess.run([train_step, global_step], feed_dict=train_feed)
            local_step += 1

            now = time.time()
            print('%f: Worker %d: traing step %d dome '
                  '(global step:%d)' % (now, FLAGS.task_index, local_step, step))

            if step >= FLAGS.train_steps:
                    break

        time_end = time.time()
        print('Training ends @ %f' % time_end)
        train_time = time_end - time_begin
        print('Training elapsed time:%f s' % train_time)

        val_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        val_xent = sess.run(cross_entropy, feed_dict=val_feed)
        print('After %d training step(s), 交叉熵损失值'
                  '= %g' % (FLAGS.train_steps, val_xent))
        sess.close()
        # sv.stop()

if __name__ == '__main__':
    tf.app.run()  #默认是main函数，如果是其他名字，使用tf.app.run(test())


