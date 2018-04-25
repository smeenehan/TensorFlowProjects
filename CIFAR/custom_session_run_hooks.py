import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.training import training_util
from tensorflow.python.training.session_run_hook import SessionRunArgs

class CustomCheckpointSaverHook(tf.train.CheckpointSaverHook):
  """Because I don't want to use the default CheckpointSaverHook behavior, which
  appends the full graph definition in the summary events file every checkpoint, 
  I'm writing a custom sub-class which overrides the offending function."""

  def __init__(self, checkpoint_dir, save_secs=None, save_steps=None, saver=None,
               checkpoint_basename="model.ckpt", scaffold=None, listeners=None):
    super().__init__(checkpoint_dir, save_secs=save_secs, save_steps=save_steps,
                     saver=saver, checkpoint_basename=checkpoint_basename,
                     scaffold=scaffold, listeners=listeners)

  def before_run(self, run_context):
    """Essentially a copy of before_run as defined in the base class, except we
    don't add the default graph or any meta-graph data to the SummaryWriter"""
    if self._timer.last_triggered_step() is None:
      training_util.write_graph(
          ops.get_default_graph().as_graph_def(add_shapes=True),
          self._checkpoint_dir,
          "graph.pbtxt")
      saver_def = self._get_saver().saver_def if self._get_saver() else None

    return SessionRunArgs(self._global_step_tensor)