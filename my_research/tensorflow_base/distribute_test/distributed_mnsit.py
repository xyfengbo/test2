
import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data

# cluster specification
parameter_servers = ["14.29.234.36:2222"]
workers = [ "14.29.234.108:2222",
            "14.29.234.224:2222"]

cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})

# input flags
tf.app.flags.DEFINE_string("job_name", "", "'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer('train_steps', 20000, 'Number of training steps to perform')#循环次数
FLAGS = tf.app.flags.FLAGS

# start a server for a specific task
server = tf.train.Server(cluster,job_name=FLAGS.job_name,task_index=FLAGS.task_index)

# config
batch_size = 100
learning_rate = 0.005
logs_path = "/tmp/mnist-data"
frequency = 100


# load mnist data set

mnist = input_data.read_data_sets(logs_path, one_hot=True)

if FLAGS.job_name == "ps":
    print('等待终端连接状态...')
    server.join()
#如果是工作节点，则开始执行
elif FLAGS.job_name == "worker":
    # Between-graph replication
    with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_index,
            cluster=cluster)):
        #worker_device="/job:worker/task:%d" % FLAGS.task_index ,少了的话，主节点执行完毕后报错，从节点断裂
        # count the number of updates
        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)

        # input images
        with tf.name_scope('input'):
            # None -> batch size can be any size, 784 -> flattened mnist image
            x = tf.placeholder(tf.float32, shape=[None, 784], name="x-input")
            # target 10 output classes
            y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")

        # model parameters will change during training so we use tf.Variable
        tf.set_random_seed(1)
        with tf.name_scope("weights"): #100个隐层
            W1 = tf.Variable(tf.random_normal([784, 100]))
            W2 = tf.Variable(tf.random_normal([100, 10]))
        # bias
        with tf.name_scope("biases"):
            b1 = tf.Variable(tf.zeros([100]))
            b2 = tf.Variable(tf.zeros([10]))

        # implement model
        with tf.name_scope("softmax"):
            # y is our prediction
            z2 = tf.add(tf.matmul(x,W1),b1)
            a2 = tf.nn.sigmoid(z2)
            z3 = tf.add(tf.matmul(a2,W2),b2)
            y  = tf.nn.softmax(z3)

        # specify cost function
        with tf.name_scope('cross_entropy'):
            # this is our cost
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

        # specify optimizer
        with tf.name_scope('train'):
            grad_op = tf.train.GradientDescentOptimizer(learning_rate)
            train_op = grad_op.minimize(cross_entropy, global_step=global_step)

        with tf.name_scope('Accuracy'):
            # accuracy
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.contrib.deprecated.scalar_summary("cost", cross_entropy)
        tf.contrib.deprecated.scalar_summary("accuracy", accuracy)

        summary_op = tf.contrib.deprecated.merge_all_summaries()
        init_op = tf.global_variables_initializer()
        print("变量初始化完毕 ...")

    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                                        global_step=global_step,
                                                        init_op=init_op)

    tf.reset_default_graph()
    with sv.prepare_or_wait_for_session(server.target) as sess:
        '''
        config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
        '''
        start_time = time.time()
        begin_time = time.time()
        count = 0
        while True:

            batch_x, batch_y = mnist.train.next_batch(batch_size)

            # perform the operations we defined earlier on batch
            _, cost, summary, step = sess.run(
            [train_op, cross_entropy, summary_op, global_step],
                feed_dict={x: batch_x, y_: batch_y})
            count += 1
            if count % frequency == 0 :
                elapsed_time = time.time() - start_time
                start_time = time.time()
                print("Step: %d," % (step + 1),
                      " Cost: %.4f," % cost,
                      " AvgTime: %3.2fms" % float(elapsed_time * 1000 / frequency))
            if step >= FLAGS.train_steps:
                    break
        print("Test-Accuracy: %2.2f" % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
        print("Total Time: %3.2fs" % float(time.time() - begin_time))
        print("Final Cost: %.4f" % cost)
    sv.stop()
    print("完毕！")
