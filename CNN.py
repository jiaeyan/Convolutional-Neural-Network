import tensorflow as tflow

class CNN():
    '''
    This is a CNN model to classify sentence relation sense.
    
    @Parameters:
        sen_len: the total length of two concatenate sentences
        emb_len: the embedding dimension
        num_class: the number of labels of input data
        vocab_len: the vocabulary size of input data
        filter_sizes: a list of filter sizes used in the model
        num_filter: the number of one size filter
    '''  
    def __init__(self, sen_len,
                       emb_len,
                       num_class,
                       vocab_len,
                       filter_sizes,
                       num_filter):
        self.sen_len = sen_len
        self.emb_len = emb_len
        self.num_class = num_class
        self.vocab_len = vocab_len
        self.filter_sizes = filter_sizes
        self.num_filter = num_filter
        
        self.emb_layer()
        pooled_outputs = self.con_pool_layer()
        self.softmax_layer(pooled_outputs)
        
    # Define an embedding layer
    def emb_layer(self):
        self.input_x = tf.placeholder(tf.int32, [None, self.sen_len])
        self.input_y = tf.placeholder(tf.float32, [None, self.num_class])
        self.dropout_keep_prob = tf.placeholder(tf.float32)
        
        EmbMatrix = tf.Variable(tf.random_uniform([self.vocab_len, self.emb_len], -1.0, 1.0))
        self.embedded_chars = tf.nn.embedding_lookup(EmbMatrix, self.input_x)
        self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
    
    # Define convolutional and max pooling layers
    def con_pool_layer(self):
        pooled_outputs = []
        for filter_size in self.filter_sizes:
            # Convolutional layer
            filter_shape = [filter_size, self.emb_len, 1, self.num_filter]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=[self.num_filter]))
            conv = tf.nn.conv2d(
                self.embedded_chars_expanded,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            
            # Max-pooling layer
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, self.sen_len - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")
            pooled_outputs.append(pooled)
        return pooled_outputs
    
    # Define the final softmax layer
    def softmax_layer(self, pooled_outputs):
        # Combine all the pooled features
        num_filters_total = self.num_filter * len(self.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        
        # Add dropout
        self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
        
        #Scores and predictions for sense
        W = tf.Variable(tf.truncated_normal([num_filters_total, self.num_class], stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[self.num_class]))
        self.scores = tf.nn.xw_plus_b(self.h_drop, W, b)
        self.predictions = tf.argmax(self.scores, 1)
        
        # Calculate mean cross-entropy loss for sense
        losses = tf.nn.softmax_cross_entropy_with_logits(logits = self.scores, labels = self.input_y)
        self.loss = tf.reduce_mean(losses)
        
        # Calculate Accuracy for sense
        correct_preds = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_preds, "float"))   
        
