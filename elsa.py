from utils import *
import tensorflow as tf
import tensorflow_addons as tfa

class ELSA_Layer(tf.keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(ELSA_Layer, self).__init__()
        
        w_init = tf.keras.initializers.HeNormal()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, units), dtype="float32"),
            trainable=True,
        )
    
    def get_items_embeddings(self):
        return self.w.numpy()
    
    def get_users_embeddings(self, users_interactions):
        embeddings = tf.math.l2_normalize(self.w, axis=-1) # numi_items, 256 (256 is d, the size of item embeddings)
        ET = tf.transpose(embeddings)
        
        ETE = ET@embeddings # 256, 256
        ETE_1 = tf.linalg.inv(ETE) # 256, 256
        ETE_1_ET = ETE_1@ET # 256, num_items
        
        user_embeddings = tf.matmul(ETE_1_ET, users_interactions, transpose_b=True)
        return user_embeddings.numpy()
    
    @tf.function
    def call(self, x):
        # A
        feature = tf.math.l2_normalize(self.w, axis=-1)
        # sum(A*A)
        #diagonal = tf.reduce_sum(feature*feature, axis =-1)
        # xA
        ret = tf.matmul(x, feature, transpose_b=False)
        # xAA^T
        ret = tf.matmul(ret, feature, transpose_b=True)
        # xAA^T-sum(A*A)
        out = ret - x
        return tf.nn.relu(out)

class ELSA(Model):
    class Model(tf.keras.Model):
        def __init__(self, latent, num_words):
            """
            num_words             nr of items in dataset (size of tokenizer)
            latent                size of latent space
            """
            super(ELSA.Model, self).__init__()     
            self.elsa = ELSA_Layer(latent, num_words)
        def call(self, x, training=None):     
            d1 = self.elsa(x)
            return d1
        
    def create_model(self, latent=128, summary=False):
        self.model = ELSA.Model(latent=latent, num_words=self.split.train_gen[0][0].shape[1])
        self.model(self.split.train_gen[0][0])
        if summary:
            self.model.summary()
        self.mc = MetricsCallback(self)

    def compile_model(self, lr=0.1):
        """
        lr         learning rate of Nadam optimizer
        fl_alpha   alpha parameter of focal crossentropy
        fl_gamma   gamma parameter of focal crossentropy
        """
        self.model.compile(
            optimizer=tf.keras.optimizers.Nadam(lr),
            loss=cosine_loss,
            metrics=['mse', cosine_loss]
        )

    def train_model(self, epochs=10):
        self.model.fit(
            self.split.train_gen,
            validation_data=self.split.validation_gen,
            epochs=epochs,
            callbacks=[self.mc]
        )

