import tensorflow as tf
import tensorflow.contrib.eager as tfe
import time

class ModelTrainer(object):

    def __init__(self, model, loss, optimizer):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.step_counter = tf.train.get_or_create_global_step()

    def train(self, dataset, num_epochs=1, val=None):
        for idx in range(num_epochs):
            start = time.time()
            if tf.executing_eagerly():
                self._train_eager_one_epoch(dataset)
            end = time.time()
            print('\nTrain time for epoch #%d: %f' % (idx+1,end - start))
            if val is not None:
                val_loss = self.eval(val)
                print('\nValidation loss for epoch #%d: %f' % (idx+1,val_loss))

    def _train_eager_one_epoch(self, dataset):
        for (batch, (features, labels)) in enumerate(tfe.Iterator(dataset)):
            with tfe.GradientTape() as tape:
                logits = self.model(features, training=True)
                loss_value = self.loss(labels, logits)
            grads = tape.gradient(loss_value, self.model.variables)
            self.optimizer.apply_gradients(zip(grads, self.model.variables), 
                                           global_step=self.step_counter)
            if batch % 50 == 0:
                print('Step #%d \tLoss: %.6f' % (batch, loss_value))

    def eval(self, dataset):
        avg_loss = tfe.metrics.Mean('loss')
        for (features, labels) in tfe.Iterator(dataset):
            logits = self.model(features, training=False)
            avg_loss(self.loss(logits, labels))
        return avg_loss.result()
