# Author: Shiyang Jia

import os
import time
import argparse
import random
import numpy as np
import tensorflow as tf
from six.moves import xrange

from dataloader import Dataloader, load_real_image, get_mean_std
from model import HairModel
from viz import visualize, visualize_real

parser = argparse.ArgumentParser(description='My HairNet =w=')

parser.add_argument('--mode',          type=str,   default='train')
parser.add_argument('--data_dir',      type=str,   default='../data')
parser.add_argument('--epochs',        type=int,   default=1)
parser.add_argument('--batch_size',    type=int,   default=16)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--output_dir',    type=str,   default='../experiments')
parser.add_argument('--load_model',    action='store_true')

args = parser.parse_args()


def create_model(session):
    """
    Create model and initialize it or load its parameters in a session

    Args:
        session: tensorflow session
    Return:
        model: HairNet model (created / loaded)
    """
    model = HairModel(args.learning_rate, args.epochs, os.path.join(args.output_dir, 'summary'))

    if args.mode == 'train' and not args.load_model:
        print("Creating model with fresh parameters")
        session.run(tf.global_variables_initializer())
        return model

    # load a previously saved model
    # ckpt_path = os.path.join(args.output_dir, 'ckpt')
    # ckpt = tf.train.latest_checkpoint(ckpt_path)
    ckpt = '../experiments/ckpt/ckpt-7840'
    if ckpt:
        print('loading model {}'.format(os.path.basename(ckpt)))
        model.saver.restore(session, ckpt)
        return model
    else:
        raise(ValueError, 'can NOT find model')


def evaluate_pos(pos, pos_gt, weight):
    """
    compute the Euclidean distance error for one hairstyle

    Args:
        pos: [32, 32, 300] output position
        pos_gt: [32, 32, 300] position ground truth
        weight: [32, 32, 100] whether the point is visible
    Return:
        average position error
    """
    # recover position with mean and std
    pos_mean, pos_std, _, _ = get_mean_std(args.data_dir)
    pos_gt = pos_gt * pos_std + pos_mean
    pos = pos * pos_std + pos_mean

    square_loss = np.square(pos - pos_gt)
    error = np.zeros((32, 32, 100))  # Euclidean distance error
    for i in range(100):
        error[..., i] = np.sqrt(np.sum(square_loss[..., 3*i:3*i+3], axis=2))
    # compute visible error
    visible = np.where(weight > 9, 1, 0)
    visible_err = visible * error
    total_err = np.sum(visible_err) / np.sum(visible)

    return total_err


def train():
    """ just training """

    # load data
    print("loading test data")
    dataloader = Dataloader(args.data_dir, args.batch_size)
    test_x, test_y, angles = dataloader.get_test_data()
    n_tests = test_x.shape[0]
    n_samples = 1800
    n_batches = n_samples // args.batch_size
    print('total number of samples: {}'.format(n_samples))
    print('total number of steps: {}'.format(n_batches * args.epochs))

    config = tf.ConfigProto(device_count={'GPU': 1}, allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        model = create_model(sess)

        log_every_n_batches = 10
        start_time = time.time()
        # GO !!
        for e in xrange(args.epochs):
            print('working on epoch {0}/{1}'.format(e + 1, args.epochs))
            epoch_start_time = time.time()
            epoch_loss, batch_loss = 0, 0
            dataloader.flesh_batch_order()

            for i in xrange(n_batches):
                if (i+1) % log_every_n_batches == 0:
                    print('working on epoch {0}, batch {1}/{2}'.format(e+1, i+1, n_batches))
                enc_in, dec_out = dataloader.get_train_batch(i)
                _, _, step_loss, summary = model.step(sess, enc_in, dec_out, True)
                epoch_loss += step_loss
                batch_loss += step_loss
                model.train_writer.add_summary(summary, model.global_step.eval())
                if (i+1) % log_every_n_batches == 0:
                    print('current batch loss: {:.2f}'.format(batch_loss / log_every_n_batches))
                    batch_loss = 0

            epoch_time = time.time() - epoch_start_time
            print('epoch {0}/{1} finish in {2:.2f} s'.format(e+1, args.epochs, epoch_time))
            print('average epoch loss: {:.4f}'.format(epoch_loss / n_batches))

            print('saving model...')
            model.saver.save(sess, os.path.join(args.output_dir, 'ckpt'), model.global_step.eval())

            # test after each epoch
            loss, pos_err = 0, 0
            for j in range(n_tests):
                enc_in, dec_out = np.expand_dims(test_x[j], 0), np.expand_dims(test_y[j], 0)  # input must be [?, 32, 32, 500]
                pos, curv, step_loss = model.step(sess, enc_in, dec_out, False)
                step_pos_err = evaluate_pos(pos[0], test_y[j, ..., 100:400], test_y[j, ..., :100])
                loss += step_loss
                pos_err += step_pos_err
            avg_pos_err = pos_err / n_tests
            err_summary = sess.run(model.err_m_summary, {model.err_m: avg_pos_err})
            model.test_writer.add_summary(err_summary, model.global_step.eval())

            print('=================================\n'
                  'total loss avg:            %.4f\n'
                  'position error avg(m):     %.4f\n'
                  '=================================' % (loss / n_tests, avg_pos_err))
            #pos = reconstruction(pos, curv)
            #avg_pos_err = evaluate_pos(pos, test_y[..., 100:400], test_y[..., :100])
            #print('position error after reconstruction: %.4e' % avg_pos_err)

        print('training finish in {:.2f} s'.format(time.time() - start_time))


def sample():
    """ test on all test data and random sample 1 hair to visualize """

    dataloader = Dataloader(args.data_dir, 0)
    test_x, test_y, angles = dataloader.get_test_data()
    n_tests = test_y.shape[0]

    # random permutation
    # order = np.random.permutation(test_y.shape[0])
    # test_x, test_y, angles = test_x[order], test_y[order], angles[order]

    config = tf.ConfigProto(device_count={'GPU': 1}, allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        model = create_model(sess)
        # start testing
        loss, pos_err = 0, 0
        best_loss, best_err = 10, 10
        idx = 0
        for i in range(n_tests):
            enc_in, dec_out = np.expand_dims(test_x[i], 0), np.expand_dims(test_y[i], 0)  # input must be [?, 32, 32, 500]
            pos, curv, step_loss = model.step(sess, enc_in, dec_out, False)
            step_pos_err = evaluate_pos(pos[0], test_y[i, ..., 100:400], test_y[i, ..., :100])
            loss += step_loss
            pos_err += step_pos_err
            if step_loss < best_loss:
                idx = i
                best_loss = step_loss
                best_err = step_pos_err
                best_pos = pos

    # re_pos = reconstruction(pos, curv)
    # pos_re_err = evaluate_pos(re_pos, test_y[..., 100:400], test_y[..., :100])
    # print('position error after reconstruction: %.4e' % pos_re_err)
        print('==================================\n'
              'total loss avg:            %.4f\n'
              'position error avg(m):     %.4f\n'
              '==================================' % (loss / n_tests, pos_err / n_tests))
        print('best')
        print('==================================\n'
              'total loss avg:            %.4f\n'
              'position error avg(m):     %.4f\n'
              '==================================' % (best_loss, best_err))
    # choose the last one
    visualize(args.data_dir, test_x[idx], test_y[idx, ..., 100:400], best_pos[0], angles[idx])


def demo():
    """use real image to inference"""

    print("loading data")
    test_data = load_real_image(os.path.join(args.data_dir, 'real'))
    test_data = test_data[0]
    cheat_y = np.zeros((1, 32, 32, 500))

    config = tf.ConfigProto(device_count={'GPU': 1}, allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        model = create_model(sess)
        print('start testing')
        pos, curv, _, _, _ = model.step(sess, test_data, cheat_y, False)
    # pos = reconstruction(pos, curv)
    visualize_real(args.data_dir, test_data, pos)


def main():
    if args.mode == 'train':
        train()
    if args.mode == 'sample':
        sample()
    if args.mode == 'demo':
        demo()


if __name__ == '__main__':
    main()
