from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
import os
import tensorflow as tf
from keras import backend as k



parameter_servers = ["14.29.234.36:2222"]
workers = ["14.29.234.108:2222",
           "14.29.234.224:2222"]
cluster = tf.train.ClusterSpec({"ps": parameter_servers, "worker": workers})
tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer('train_steps', 10000, 'Number of training steps to perform')  # 循环次数
FLAGS = tf.app.flags.FLAGS
server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
# server = tf.train.Server.create_local_server()
sess = tf.Session(server.target)
tf.get_default_graph()
k.set_session(sess)



def keras_proc():
    '''
    keras处理主要部分
    :return:
    '''
    # define documents
    docs = ['Well done!',
            'Good work',
            'Great effort',
            'nice work',
            'Excellent!',
            'Weak',
            'Poor effort!',
            'not good',
            'poor work',
            'Could have done better.']
    # define class labels
    labels = array([1,1,1,1,1,0,0,0,0,0])
    # integer encode the documents
    vocab_size = 50
    encoded_docs = [one_hot(d, vocab_size) for d in docs]
    max_length = 4
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    # define the model
    model = Sequential()
    model.add(Embedding(vocab_size, 8, input_length=max_length))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    # summarize the model
    print(model.summary())
    # fit the model
    model.fit(padded_docs, labels, epochs=50000, verbose=0)
    # evaluate the model
    loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
    print('Accuracy: %f' % (accuracy*100))

def main():
    keras_proc()


def main_dist():
    '''
    测试函数入口
    :return:
    '''
    # KTF.set_session(getSession())
    # session_init()
    if FLAGS.job_name == "ps":
        print('等待终端连接状态...')
        server.join()
    elif FLAGS.job_name == "worker":
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):


            global_step = tf.get_variable('global_step', [],
                                          initializer=tf.constant_initializer(0),
                                          trainable=False)
            init_op = tf.global_variables_initializer()
            print("初始化变量")
        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                 global_step=global_step,
                                 init_op=init_op)
        tf.reset_default_graph()
        with sv.prepare_or_wait_for_session(server.target) as sess:
            count = 0 #循环计数器
            while True:
                # sess.run(init_op)
                step = sess.run(global_step)
                keras_proc()
                print('step=%d'%step)
                count += 1
                if step >= FLAGS.train_steps: #循环次数大于指定的次数，退出
                    break
        k.clear_session()
        sv.stop()



if __name__=='__main__':
    main()
