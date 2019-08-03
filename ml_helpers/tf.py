from .imports import *
from tensorflow.python.data import Dataset
import tensorflow as tf

class StopWhenAccReachedCallback(tf.keras.callbacks.Callback):
  def __init__(self, acc=None, loss=None):
    self.acc = acc
    self.loss = loss
  """
  callback = StopWhenAccReachedCallback(0.99)
  model.fit(X_train, Y_train, batch_size=batch_size, epochs=20, callbacks = [callback])
  """
  def on_epoch_end(self, epoch, logs={}):
      if self.acc and logs.get('acc') < self.acc:
          print(f"\nAccuracy is {self.acc}, cancelling training")
          self.model.stop_training = True
      if self.loss and logs.get('loss') < self.loss:
          print(f"\nLoss is {self.loss}, cancelling training")
          self.model.stop_training = True

def input_func(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Constructs an input function to put into a Tensorflow Estimator
    Source: https://colab.research.google.com/notebooks/mlcc/first_steps_with_tensor_flow.ipynb#scrollTo=RKZ9zNcHJtwc
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
  
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}                                           
 
    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(buffer_size=10000)
    
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

def visualize_activations(successive_feature_maps, layer_names):
  """
  Source: https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%202%20-%20Part%202%20-%20Lesson%202%20-%20Notebook.ipynb#scrollTo=tuqK2arJL0wo
  """
  for layer_name, feature_map in zip(layer_names, successive_feature_maps):
  
    # Just do this for the conv / maxpool layers
    if len(feature_map.shape) == 4:
        n_features = feature_map.shape[-1]
        size = feature_map.shape[ 1]

        # We will tile our images in this matrix
        display_grid = np.zeros((size, size * n_features))

        #-------------------------------------------------
        # Postprocess the feature to be visually palatable
        #-------------------------------------------------
        for i in range(n_features):
            x  = feature_map[0, :, :, i]
            x -= x.mean()
            x /= x.std ()
            x *=  64
            x += 128
            x  = np.clip(x, 0, 255).astype('uint8')
            display_grid[:, i * size : (i + 1) * size] = x # Tile each filter into a horizontal grid

        scale = 100. / n_features
        plt.figure( figsize=(scale * n_features, scale) )
        plt.title ( layer_name )
        plt.grid  ( False )
        plt.imshow( display_grid, aspect='auto', cmap='viridis' ) 