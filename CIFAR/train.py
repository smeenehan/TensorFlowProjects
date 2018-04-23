import tensorflow as tf
import tensorflow.contrib.eager as tfe
import tensorflow.contrib.summary as summary
import time

class ModelTrainer(object):
    """Organize training of a TensorFlow model.

    Supports both eager and graph execution. Currently graph execution will
    use the default session (e.g., tf.get_default_session()).

    Attributes
    ----------
    model : function
        TensorFlow model function. Should take as input a data tensor and
        optional bool training.
    accuracy : operation
        Operation taking labels and predictions and returning the fractional
        accuracy.
    loss : operation
        Operation taking labels and predictions and returning the average loss.
    logits : operation
        Predictions of the model. Only defined in graph execution mode.
    optimizer : operation
        TensorFlow optimizer.

    Parameters
    ----------
    train_data : dataset
        Training data and labels. Should conform to the TensorFlow dataset
        API and return tuples of (features, labels) upon iteration.
    val_data : dataset
        Validation data and labels.
    test_data : dataset
        Testing data and labels.
    """
    def __init__(self, model, accuracy, loss, optimizer, train_data, val_data, 
                 test_data, summary_dir='summaries', summary_name=''):
        self.model = model
        self.data = {'train': train_data, 'val': val_data, 'test': test_data}
        self.step_counter = tf.train.get_or_create_global_step()
        self.summary = summary.create_file_writer(
            summary_dir, flush_millis=10000, filename_suffix=summary_name)
        if tf.executing_eagerly():
            self.loss = loss
            self.accuracy = accuracy
            self.optimizer = optimizer
        else:
            self._setup_graph_training(accuracy, loss, optimizer)

    def _setup_graph_training(self, accuracy, loss, optimizer):
        data_types, data_shapes = self.data['train'].output_types, \
                                  self.data['train'].output_shapes
        self._iterator = tf.data.Iterator.from_structure(data_types, data_shapes)
        self._inits = {'train': self._iterator.make_initializer(self.data['train']),
                       'val': self._iterator.make_initializer(self.data['val']),
                       'test': self._iterator.make_initializer(self.data['test'])}
        features, labels = self._iterator.get_next()
        training = tf.placeholder(tf.bool, name='training')

        self.logits = self.model(features, training=training)
        self.accuracy = accuracy(labels, self.logits)
        self.loss = loss(labels, self.logits)
        with self.summary.as_default(), summary.always_record_summaries():
            self._summary_ops = {
            'train_loss': summary.scalar('train_loss', self.loss),
            'train_accuracy': summary.scalar('train_accuracy', self.accuracy),
            'val_loss': summary.scalar('val_loss',
                                       tf.placeholder(tf.float32, name='avg_loss')),
            'val_accuracy': summary.scalar('val_accuracy', 
                                           tf.placeholder(tf.float32, name='avg_accuracy'))}
        # Needed for BatchNorm to work
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            self.optimizer = optimizer.minimize(self.loss, 
                                                global_step=self.step_counter)


    def train(self, num_epochs=1, verbose=True):
        """Train the model, and log training statistics.

        Parameters
        ----------
        num_epochs : int
            Number of epochs to train for. Defaults to 1.
        """
        with self.summary.as_default(), summary.always_record_summaries():
            tf.contrib.summary.initialize()
            print()
            val_loss, val_accuracy = self.evaluate('val')
            if verbose:
                 print('\nInitial validation loss, accuracy: %f, %f' % 
                      (val_loss, val_accuracy))

            for idx in range(num_epochs):
                start = time.time()
                if tf.executing_eagerly():
                    self._train_eager_one_epoch()
                else:
                    self._train_graph_one_epoch()
                end = time.time()
                val_loss, val_accuracy = self.evaluate('val')
                if verbose:
                    print('\nTrain time for epoch #%d: %f' % (idx+1,end - start))
                    print('\nValidation loss, accuracy for epoch #%d: %f, %f' % 
                          (idx+1,val_loss, val_accuracy))

    def _train_eager_one_epoch(self):
        for (batch, (features, labels)) in enumerate(tfe.Iterator(self.data['train'])):
            with tfe.GradientTape() as tape:
                logits = self.model(features, training=True)
                train_loss = self.loss(labels, logits)
                train_accuracy = self.accuracy(labels, logits)
            grads = tape.gradient(train_loss, self.model.variables)
            self.optimizer.apply_gradients(zip(grads, self.model.variables), 
                                           global_step=self.step_counter)
            summary.scalar('train_loss', train_loss)
            summary.scalar('train_accuracy', train_accuracy)

    def _train_graph_one_epoch(self):
        sess = tf.get_default_session()
        sess.run(self._inits['train'])
        train_ops = [self.loss, self.accuracy, self.optimizer,
                     self._summary_ops['train_loss'], 
                     self._summary_ops['train_accuracy']],
        while True:
            try:
                sess.run(train_ops, feed_dict={'training:0': True})
            except tf.errors.OutOfRangeError:
                break

    def evaluate(self, data_name):
        """Evaluate training loss on as specified dataset.

        Parameters
        ----------
        data_name : string
            Name of the data set ('train', 'val', or 'test') on which we wish to
            evaluate the model

        Returns
        -------
        float, float
            Average loss, accuracy over the full dataset.

        Raises
        ------
        ValueError
            If data_name is not valid.
        """
        if data_name not in self.data.keys():
            raise ValueError('data_name must be one of: ', self.data.keys())

        if tf.executing_eagerly():
            return self._evaluate_eager(data_name)
        else:
            return self._evaluate_graph(data_name)

    def _evaluate_eager(self, data_name):
        loss_metric = tfe.metrics.Mean('loss')
        accuracy_metric = tfe.metrics.Mean('loss')
        for (features, labels) in tfe.Iterator(self.data[data_name]):
            logits = self.model(features, training=False)
            loss_metric(self.loss(labels, logits))
            accuracy_metric(self.accuracy(labels, logits))
        avg_loss, avg_accuracy = loss_metric.result(), accuracy_metric.result()
        if data_name is 'val':
            summary.scalar('val_loss', avg_loss)
            summary.scalar('val_accuracy', avg_accuracy)
        return avg_loss, avg_accuracy

    def _evaluate_graph(self, data_name):
        sess = tf.get_default_session()
        sess.run(self._inits[data_name])
        loss_metric, accuracy_metric, cnt = 0, 0, 0
        eval_ops = [self.loss, self.accuracy]
        while True:
            try:
                loss, accuracy = sess.run(eval_ops, 
                                              feed_dict={'training:0': False})
                loss_metric += loss
                accuracy_metric += accuracy
                cnt += 1
            except tf.errors.OutOfRangeError:
                break
        avg_loss, avg_accuracy = loss_metric/cnt, accuracy_metric/cnt
        if data_name is 'val':
            sess.run([self._summary_ops['val_loss'], 
                      self._summary_ops['val_accuracy']],
                     feed_dict={'avg_loss:0': avg_loss, 
                                'avg_accuracy:0': avg_accuracy})
        return avg_loss, avg_accuracy