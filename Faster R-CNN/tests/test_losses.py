from network.losses import bbox_loss, class_loss
import numpy as np
import tensorflow as tf

class TestLosses(tf.test.TestCase):
    def test_class_loss(self):
        anchor_logits = tf.convert_to_tensor(
            np.tile([[[-0.5, 0.2]]], [2, 8, 1]).astype('float32'))
        target_classes = tf.convert_to_tensor(
            np.array([[1, 0, 1, -1, 0, 1, -1, 0], 
                      [0, 1, 1, 0, -1, 0, -1, 1]]).astype('int32'))
        loss_0 = -np.log(np.exp(-0.5)/(np.exp(-0.5)+np.exp(0.2)))
        loss_1 = -np.log(np.exp(0.2)/(np.exp(-0.5)+np.exp(0.2)))
        total_loss = 0.5*(loss_0+loss_1)

        loss = class_loss(anchor_logits, target_classes)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            loss_result = sess.run(loss)
            self.assertAlmostEqual(loss_result, total_loss)

    def test_bbox_loss(self):
        anchor_deltas = tf.convert_to_tensor(
            np.tile([[[0.05, 0.1, 0.2, -0.3], 
                       [0.12, 0.43, -1.6, 0.3]]], [2, 4, 1]).astype('float32'))
        target_classes = tf.convert_to_tensor(
            np.array([[1, 0, 1, -1, 0, 1, -1, 0], 
                      [0, 1, 1, 0, -1, 0, -1, 1]]).astype('int32'))
        target_deltas = tf.convert_to_tensor(
            np.tile([[[0.1, 0.2, 0.4, 0.6], 
                       [0.6, 0.2, 0.4, 0.5]]], [2, 4, 1]).astype('float32'))

        loss_0 = 0.5*(0.05**2+0.1**2+0.2**2+0.9**2)
        loss_1 = 0.5*(0.48**2+0.23**2+0.2**2)+1.5        
        total_loss = 3*(loss_0+loss_1)/16

        loss = bbox_loss(anchor_deltas, target_classes, target_deltas)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            loss_result = sess.run(loss)
            self.assertAlmostEqual(loss_result, total_loss)