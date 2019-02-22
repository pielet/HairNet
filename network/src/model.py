# Author: Shiyang Jia

import tensorflow as tf
import tensorflow.contrib.slim as slim


class HairModel(object):
    """
    a simple encoder-decoder model

    HairModel variables
        global_step         current global step
        learn_rate          current learn rate
        is_training         [placeholder] - training flag
        encoder_input       [placeholder] - input x
        decoder_output      [placeholder] - input y
    """

    def __init__(self, learning_rate, epochs, output_dir):
        """
        Args:
            mode            'train' or 'test'
            learn_rate      initial learn rate
            epochs          number of total epochs
            output_dir      save summary
        """
        self.is_training = tf.placeholder(tf.bool, name='is_training_flag')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # learn rate
        # boundaries = [epochs // 2]
        # values = [learning_rate, learning_rate / 2]
        # self.learn_rate = tf.train.piecewise_constant(self.global_step, boundaries, values)
        self.learn_rate = learning_rate
        
        self.train_writer = tf.summary.FileWriter(output_dir)
        self.test_writer = tf.summary.FileWriter(output_dir)

        self.build_model()
        self.build_losses()
        self.build_summaries()
        self.updates = tf.train.AdamOptimizer(self.learn_rate)\
            .minimize(self.total_loss, self.global_step)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def conv(self, x, num_out_layers, kernel_size, stride, activation_fn=tf.nn.relu):
        p = (kernel_size - stride) // 2
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn)

    def maxpool(self, x, kernel_size, activation_fn=tf.nn.tanh):  # maxpool + tanh
        max_pool = slim.max_pool2d(x, kernel_size, kernel_size)
        return activation_fn(max_pool)

    def linear(self, x, num_output, activation_fn=tf.nn.relu):
        return slim.fully_connected(x, num_output, activation_fn)

    def upsample(self, x, ratio):
        s = tf.shape(x)
        h, w = s[1], s[2]   # [batches, h, w, channel]
        return tf.image.resize_bilinear(x, [h*ratio, w*ratio])

    def build_model(self):
            enc_in = tf.placeholder(tf.float64, shape=[None, 256, 256, 2], name='enc_in')
            dec_out = tf.placeholder(tf.float32, shape=[None, 32, 32, 500], name='dec_out')
            self.encoder_input = enc_in
            self.decoder_output = dec_out

            # encoder
            with tf.variable_scope('encoder'):
                conv = self.conv(enc_in, 32,  8, 2)    # (128, 128, 32)
                conv = self.conv(conv,   64,  8, 2)    # (64, 64, 64)
                conv = self.conv(conv,   128, 6, 2)    # (32, 32, 128)
                conv = self.conv(conv,   256, 4, 2)    # (16, 16, 256)
                conv = self.conv(conv,   256, 3, 1)    # (16, 16, 256)
                conv = self.conv(conv,   512, 4, 2)    # (8, 8, 512)
                conv = self.conv(conv,   512, 3, 1)    # (8, 8, 512)
                max_pool = self.maxpool(conv, 8)       # (1, 1, 512)

            max_pool = tf.reshape(max_pool, [-1, 512])     # (512)

            # decoder
            with tf.variable_scope('common_decoder'):
                linear = self.linear(max_pool, 1024)   # (1024)
                linear = self.linear(linear,  4096)   # (4096)
                n = tf.shape(linear)[0]
                rua = tf.reshape(linear, [n, 4, 4, 256]) # (4, 4, 256)
                up   = self.upsample(rua, 2)      # (8, 8, 256)
                conv = self.conv(up, 512, 3, 1)   # (8, 8, 512)
                up   = self.upsample(conv, 2)     # (16, 16, 512)
                conv = self.conv(up, 512, 3, 1)   # (16, 16, 512)
                up   = self.upsample(conv, 2)     # (32, 32, 512)
                conv = self.conv(up, 512, 3, 1)   # (32, 32, 512)

            with tf.variable_scope('pos_decoder'):
                pos_conv = self.conv(conv,     512, 1, 1)
                pos_conv = self.conv(pos_conv, 512, 1, 1, tf.nn.tanh)
                pos_conv = self.conv(pos_conv, 300, 1, 1, None)     # (32, 32, 300)

            with tf.variable_scope('curv_decoder'):
                curv_conv = self.conv(conv,      512, 1, 1)
                curv_conv = self.conv(curv_conv, 512, 1, 1, tf.nn.tanh)
                curv_conv = self.conv(curv_conv, 100, 1, 1, None)   # (32, 32, 100)

            self.pos_output = pos_conv
            self.curv_output = curv_conv

    def build_losses(self):
        weight  = self.decoder_output[..., :100]
        pos_gt  = self.decoder_output[..., 100:400]
        curv_gt = self.decoder_output[..., 400:500]

        # position loss
        pos_weight = tf.tile(tf.reshape(weight, [-1, 1]), [1, 3])
        pos_weight = tf.reshape(pos_weight, [-1, 32, 32, 300])
        self.pos_loss = tf.reduce_mean(pos_weight * tf.square(self.pos_output - pos_gt))
        # curvature loss
        self.curv_loss = tf.reduce_mean(weight * tf.square(self.curv_output - curv_gt))

        self.total_loss = self.pos_loss + self.curv_loss

    def build_summaries(self):
        # loss summary
        self.pos_summary = tf.summary.scalar('pos_loss', self.pos_loss)
        self.curv_summary = tf.summary.scalar('curv_loss', self.curv_loss)
        self.total_loss_summary = tf.summary.scalar('total_loss', self.total_loss)
        self.loss_summary = tf.summary.merge([self.pos_summary,
                                              self.curv_summary,
                                              self.total_loss_summary])
        # error summary
        self.err_m = tf.placeholder(tf.float64, name='error_m')
        self.err_m_summary = tf.summary.scalar('error_m', self.err_m)

    def step(self, session, encoder_input, decoder_output, is_training):
        """
        run a step of the model feeding the given inputs
        Args:
            session: tensorflow session to use
            is_training: train or test
            encoder_input: color-coded orientation map
            decoder_output: hair position and curvatures ground truth
        Return:
            traning: position, curvatures, total_loss, loss_summary
            testing: position, curvatures, total_loss
        """
        input_feed = {self.is_training: is_training,
                      self.encoder_input:  encoder_input,
                      self.decoder_output: decoder_output}

        # output feed: depends on whether in training mode
        if is_training:
            output_feed = [self.pos_output,
                           self.curv_output,
                           self.total_loss,
                           self.updates,
                           self.loss_summary]
            outputs = session.run(output_feed, input_feed)
            return outputs[0], outputs[1], outputs[2], outputs[4]
        else:
            output_feed = [self.pos_output,
                           self.curv_output,
                           self.total_loss]
            outputs = session.run(output_feed, input_feed)
            return outputs[0], outputs[1], outputs[2]
