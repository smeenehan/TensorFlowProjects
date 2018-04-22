import tensorflow as tf
import tensorflow.contrib.eager as tfe
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
    loss : operation
        Loss operation. Should take labels and predictions and return the 
        average batch loss.
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
    def __init__(self, model, loss, optimizer, train_data, val_data, test_data):
        self.model = model
        self.data = {'train': train_data, 'val': val_data, 'test': test_data}
        if tf.executing_eagerly():
            self.loss = loss
            self.optimizer = optimizer
        else:
            self._setup_graph_training(loss, optimizer)
        self.step_counter = tf.train.get_or_create_global_step()

    def _setup_graph_training(self, loss, optimizer):
        data_types, data_shapes = self.data['train'].output_types, \
                                  self.data['train'].output_shapes
        self._iterator = tf.data.Iterator.from_structure(data_types, data_shapes)
        self._inits = {'train': self._iterator.make_initializer(self.data['train']),
                       'val': self._iterator.make_initializer(self.data['val']),
                       'test': self._iterator.make_initializer(self.data['test'])}
        features, labels = self._iterator.get_next()
        training = tf.placeholder(tf.bool, name='training')

        self.logits = self.model(features, training=training)
        self.loss = loss(labels, self.logits)
        # Needed for BatchNorm to work
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            self.optimizer = optimizer.minimize(self.loss)


    def train(self, num_epochs=1, verbose=True):
        """Train the model, and log training statistics.

        Parameters
        ----------
        num_epochs : int
            Number of epochs to train for. Defaults to 1.
        """
        for idx in range(num_epochs):
            start = time.time()
            if tf.executing_eagerly():
                self._train_eager_one_epoch()
            else:
                self._train_graph_one_epoch()
            end = time.time()
            val_loss = self.evaluate('val')
            if verbose:
                print('\nTrain time for epoch #%d: %f' % (idx+1,end - start))
                print('\nValidation loss for epoch #%d: %f' % (idx+1,val_loss))

    def _train_eager_one_epoch(self):
        for (batch, (features, labels)) in enumerate(tfe.Iterator(self.data['train'])):
            with tfe.GradientTape() as tape:
                logits = self.model(features, training=True)
                train_loss = self.loss(labels, logits)
            grads = tape.gradient(train_loss, self.model.variables)
            self.optimizer.apply_gradients(zip(grads, self.model.variables), 
                                           global_step=self.step_counter)

    def _train_graph_one_epoch(self):
        sess = tf.get_default_session()
        sess.run(self._inits['train'])
        train_ops = [self.loss, self.optimizer]
        while True:
            try:
                train_loss, _ = sess.run(
                    train_ops, feed_dict={'training:0': True})
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
        float
            Average loss over the full dataset.

        Raises
        ------
        ValueError
            If data_name is not valid.
        """
        if data_name not in self.data.keys():
            raise ValueError('data_name must be one of: ', self.data.keys())

        if tf.executing_eagerly():
            loss = self._evaluate_eager(data_name)
        else:
            loss = self._evaluate_graph(data_name)
        return loss

    def _evaluate_eager(self, data_name):
        avg_loss = tfe.metrics.Mean('loss')
        for (features, labels) in tfe.Iterator(self.data[data_name]):
            logits = self.model(features, training=False)
            avg_loss(self.loss(labels, logits))
        return avg_loss.result()

    def _evaluate_graph(self, data_name):
        sess = tf.get_default_session()
        sess.run(self._inits[data_name])
        avg_loss, cnt = 0, 0
        while True:
            try:
                loss = sess.run(self.loss, feed_dict={"training:0": False})
                avg_loss += loss
                cnt += 1
            except tf.errors.OutOfRangeError:
                break
        avg_loss /= cnt
        return avg_loss