import tensorflow as tf

def make_feature_weighted_mse(feature_weights):
    
    feature_weights = tf.reshape(tf.cast(feature_weights, 'float32'), (-1,1))
    
    def feature_weighted_mse(y_true, y_pred):
        
        y_true = tf.cast(y_true, 'float32')
        y_pred = tf.cast(y_pred, 'float32')

        return tf.linalg.matmul(tf.square(y_true - y_pred), feature_weights)
        #tf.reduce_sum(
    
    return feature_weighted_mse
