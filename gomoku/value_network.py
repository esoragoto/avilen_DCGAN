
# coding: utf-8

# # 五目並べAI
# 
# ## 概要
# 
# AIはvalue_networkと盤面探索を組み合わせています。<br>
# value_networkは五目並べの盤面から、ゲームの情勢を判断するモデルです。<br>
# AIはvalue_networkにより次のすべての盤面の中から最も自分にとって優勢のものを選択し、打つ手を決定します。<br>

# In[4]:


import tensorflow as tf
import os


# ## CNN
# 
# value_networkはシンプルなCNNによって構成されています。<br>
# 盤面情報を画像として入力し、黒の勝率を出力します。<br>
# 
# 入力次元 - (Batch_size x size x size x channel)
# - Batch_size: 学習のミニバッチのサイズ（任意）
# - size: 盤面サイズ(デフォルト9)
# - channel: 2(黒と白)
# 
# 出力次元 - (1)

# In[7]:


size = 9


# In[5]:


def cnn(x):
    x_image = tf.reshape(x, [-1, size, size, 2])    # [None, size, size, 2]
    conv1 = tf.compat.v1.layers.conv2d(x_image, 128, (3, 3), padding='same',
                             activation=tf.nn.relu)    # [None, size, size, 128]
    conv2 = tf.compat.v1.layers.conv2d(conv1, 128, (3, 3), padding='same',
                             activation=tf.nn.relu)    # [None, size, size, 128]
    pool_flat = tf.compat.v1.layers.flatten(conv2) # [None, size * size * 128]
    dense = tf.compat.v1.layers.dense(pool_flat, 64, activation=tf.nn.relu) # [None, 64]
    y = tf.compat.v1.layers.dense(dense, 1) # [None, 1]
    return y


# In[6]:


class value_network_model:
    def __init__(self, size=9, rate=1e-6):
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.disable_eager_execution()
        self.x = tf.compat.v1.placeholder(tf.float32, (None, size, size, 2))
        self.t = tf.compat.v1.placeholder(tf.float32, (None, 1))
        self.y = cnn(self.x)
        self.trail = 0
        self.model_path = './model/'
        self.logs_path = './log/'
        self.saver = tf.compat.v1.train.Saver()
        self.cost = tf.reduce_mean(input_tensor=tf.square(self.y - self.t))
        self.optimizer = tf.compat.v1.train.AdamOptimizer(rate).minimize(self.cost)
        self.train_summary_loss = tf.compat.v1.summary.scalar('train_loss', self.cost)
        self.saver = tf.compat.v1.train.Saver()
        self.summary_writer = tf.compat.v1.summary.FileWriter(
            self.logs_path, graph=tf.compat.v1.get_default_graph())
        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        else:
            self.load()

    def load(self):
        try:
            self.saver.restore(self.sess, self.model_path)
        except:
            print('WARNING: There is nothing in the `model` directory, and `self.saver.restore()` has not been executed.')

    def new_logs(self):
        if os.path.exists(self.logs_path):
            shutil.rmtree(self.logs_path)
        os.mkdir(self.logs_path)

    # モデル予測値の出力
    def out(self, X):
        return self.y.eval(feed_dict={self.x: X}, session=self.sess)

    # 最適化
    def optimize(self, X, T):
        _, cost, summary_loss = self.sess.run(
            [self.optimizer, self.cost, self.train_summary_loss], feed_dict={self.x: X, self.t: T})
        # ログの保存
        self.summary_writer.add_summary(summary_loss, self.trail)
        self.trail += 1
        # モデルの保存
        self.saver.save(self.sess, self.model_path)
        return cost

