import tensorflow as tf
import tensorflow.contrib.eager as tfe
import time

def data_spec(dataset):
    return (dataset.output_types, dataset.output_shapes)

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
    data_spec : tuple
        Tuple of data types and shapes served by the Datasets used for training
        and evaluation. Used in graph execution to define input placeholders. 
        Defaults to None.

    Raises
    ------
    ValueError
        If data_spec is not specified in graph execution mode.
    """
    def __init__(self, model, loss, optimizer, data_spec=None):
        self.model = model
        if tf.executing_eagerly():
            self.loss = loss
            self.optimizer = optimizer
        else:
            if data_spec is None:
                raise ValueError('Must specify data spec in graph mode')
            self._setup_graph_training(loss, optimizer, data_spec)
        self.step_counter = tf.train.get_or_create_global_step()

    def _setup_graph_training(self, loss, optimizer, data_spec):
        data_types, data_shapes = data_spec
        self._iterator = tf.data.Iterator.from_structure(data_types, data_shapes)
        features, labels = self._iterator.get_next()
        training = tf.placeholder(tf.bool, name='training')

        self.logits = self.model(features, training=training)
        self.loss = loss(labels, self.logits)
        # Needed for BatchNorm to work
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            self.optimizer = optimizer.minimize(self.loss)


    def train(self, train_data, num_epochs=1, val_data=None):
        """Train the model, and log training statistics.

        Parameters
        ----------
        train_data : dataset
            Training data and labels. Should conform to the TensorFlow dataset
            API and return tuples of (features, labels) upon iteration.
        num_epochs : int
            Number of epochs to train for. Defaults to 1.
        val_data : dataset
            Validation data and labels. If present, validation loss is computed 
            at the beginning of training and after each epoch. Defaults to None.
        """
        for idx in range(num_epochs):
            start = time.time()
            if tf.executing_eagerly():
                self._train_eager_one_epoch(train_data)
            else:
                self._train_graph_one_epoch(train_data)
            end = time.time()
            print('\nTrain time for epoch #%d: %f' % (idx+1,end - start))
            if val_data is not None:
                val_loss = self.eval(val_data)
                print('\nValidation loss for epoch #%d: %f' % (idx+1,val_loss))

    def _train_eager_one_epoch(self, train_data):
        for (batch, (features, labels)) in enumerate(tfe.Iterator(train_data)):
            with tfe.GradientTape() as tape:
                logits = self.model(features, training=True)
                train_loss = self.loss(labels, logits)
            grads = tape.gradient(train_loss, self.model.variables)
            self.optimizer.apply_gradients(zip(grads, self.model.variables), 
                                           global_step=self.step_counter)
            if batch % 50 == 0:
                print('Step #%d \tLoss: %.6f' % (batch, train_loss))

    def _train_graph_one_epoch(self, train_data):
        sess = tf.get_default_session()
        train_init = self._iterator.make_initializer(train_data)

        sess.run(train_init)
        train_ops = [self.loss, self.optimizer]
        batch = 0
        while True:
            try:
                train_loss, _ = sess.run(
                    train_ops, feed_dict={'training:0': True})
                if batch % 50 == 0:
                    print('Step #%d \tLoss: %.6f' % (batch, train_loss))
                batch += 1
            except tf.errors.OutOfRangeError:
                break

    def eval(self, data):
        """Evaluate training loss on a dataset.

        Parameters
        ----------
        data : dataset
            Training data and labels. Should conform to the TensorFlow dataset
            API and return tuples of (features, labels) upon iteration.

        Returns
        -------
        float
            Average loss over the full dataset.
        """
        if tf.executing_eagerly(): 
            loss = self._eval_eager(data)
        else:
            loss = self._eval_graph(data)
        return loss

    def _eval_eager(self, data):
        avg_loss = tfe.metrics.Mean('loss')
        for (features, labels) in tfe.Iterator(data):
            logits = self.model(features, training=False)
            avg_loss(self.loss(labels, logits))
        return avg_loss.result()

    def _eval_graph(self, data):
        sess = tf.get_default_session()
        data_init = self._iterator.make_initializer(data)

        sess.run(data_init)
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