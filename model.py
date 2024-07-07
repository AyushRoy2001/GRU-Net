import tensorflow as tf
import tensorflow.keras as keras
import keras.backend as K
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, MaxPool2D, Add, Dropout, Concatenate, Conv2DTranspose, Dense, Reshape, Flatten, Softmax, Lambda, UpSampling2D, AveragePooling2D, Activation, BatchNormalization, GlobalAveragePooling2D, SeparableConv2D
from tensorflow.keras.optimizers import AdamW
import tensorflow_probability as tfp

'''texts = [
    "tumor epithelial tissue",
    "necrotic tissue",
    "lymphocytic tissue",
    "tumor associated stromal tissue",
    "coagulative necrosis",
    "liquefactive necrosis",
    "desmoplasia",
    "granular and non granular leukocytes",
    "perinuclear halo",
    "interstitial space",
    "neutrophils",
    "macrophages",
    "collagen",
    "fibronectin",
    "hyperplasia",
    "dysplasia"
]'''

text = pd.read_csv('/kaggle/input/tnbc-seg/text_labels.csv', header=None)
text = tf.convert_to_tensor(np.asarray(text), dtype=tf.float32)

text = Dense(32, activation='relu')(text)
text = Dense(32, activation='relu')(tf.transpose(text, perm=[1,0]))
text = tf.expand_dims(text, axis=0)
text = tf.expand_dims(text, axis=-1)

class DistributionModel(tf.keras.Model):
    def __init__(self, text):
        super(DistributionModel, self).__init__()
        self.mean_layer = tf.keras.layers.SeparableConv2D(filters=1, kernel_size=1, padding='same', activation='softplus')
        self.stddev_layer = tf.keras.layers.SeparableConv2D(filters=1, kernel_size=1, padding='same', activation='softplus')
        self.distribution_layer = tfp.layers.DistributionLambda(
            lambda t: tfp.distributions.Normal(loc=t[..., :1], scale=t[..., 1:])
        )
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.conv_t1 = tf.keras.layers.Conv2DTranspose(256, 2, strides=2, padding="same", activation='relu')
        self.conv_t2 = tf.keras.layers.Conv2DTranspose(256, 2, strides=2, padding="same", activation='relu')
        self.conv_t3 = tf.keras.layers.Conv2DTranspose(256, 2, strides=2, padding="same", activation='relu')
        self.conv_t4 = tf.keras.layers.Conv2DTranspose(256, 2, strides=2, padding="same", activation='relu')
        self.text = text
        self.conv_tt1 = tf.keras.layers.Conv2DTranspose(1, 2, strides=2, padding="same", activation='relu')
        self.conv_tt2 = tf.keras.layers.Conv2DTranspose(1, 2, strides=2, padding="same", activation='relu')
        self.conv_tt3 = tf.keras.layers.Conv2DTranspose(1, 2, strides=2, padding="same", activation='relu')
        self.conv_tt4 = tf.keras.layers.Conv2DTranspose(1, 2, strides=2, padding="same", activation='relu')

    def distribution(self, x, text):
        mean = self.mean_layer(x)+tf.math.reduce_mean(text, axis=[-1,-2,-3])
        stddev = tf.math.sqrt(tf.math.square(self.stddev_layer(x))+tf.math.reduce_variance(text, axis=[-1,-2,-3]))

        # Concatenate mean and standard deviation
        parameters = self.concat([mean, stddev])

        # Generate distribution
        distribution = self.distribution_layer(parameters)
        return distribution

    def distribution_attn(self, x):
        x1 = self.conv_t1(x)
        x2 = self.conv_t2(x1)
        x3 = self.conv_t3(x2)
        x4 = self.conv_t4(x3)
        text = self.conv_tt1(self.text)
        dis1 = self.distribution(x1, text)
        text = self.conv_tt2(text)
        dis2 = self.distribution(x2, text)
        text = self.conv_tt3(text)
        dis3 = self.distribution(x3, text)
        text = self.conv_tt4(text)
        dis4 = self.distribution(x4, text)
        return dis1, dis2, dis3, dis4

    def call(self, inputs):
        return self.distribution_attn(inputs)

def conv_block(x, num_filters, kernel_size, padding="same", act=True):
    x = Conv2D(num_filters, kernel_size, padding=padding, use_bias=False)(x)
    x = BatchNormalization()(x)
    if act:
        x = Activation("relu")(x)
    return x

def multires_block(x, num_filters, alpha=1.67):
    W = num_filters * alpha

    x0 = x
    x1 = conv_block(x0, int(W*0.167), 3)
    x2 = conv_block(x1, int(W*0.333), 3)
    x3 = conv_block(x2, int(W*0.5), 3)
    xc = Concatenate()([x1, x2, x3])
    xc = BatchNormalization()(xc)

    nf = int(W*0.167) + int(W*0.333) + int(W*0.5)
    sc = conv_block(x0, nf, 1, act=False)

    x = Activation("relu")(xc + sc)
    x = BatchNormalization()(x)
    return x

def res_path(x, num_filters, length): 
    check = L.GlobalMaxPooling2D()(x)
    check = L.Dense(1, activation='sigmoid')(x)
    check = tf.math.reduce_mean(check, axis=0)
    
    x01 = x
    x11 = conv_block(x01, num_filters, 3, act=False)
    sc1 = conv_block(x01, num_filters, 1, act=False)
    x = Activation("relu")(x11 + sc1)
    x = BatchNormalization()(x)
    
    x02 = Concatenate()([x,x01])
    x12 = conv_block(x02, num_filters, 3, act=False)
    sc2 = conv_block(x02, num_filters, 1, act=False)
    x = Activation("relu")(x12 + sc2)
    x = BatchNormalization()(x)
    
    x03 = Concatenate()([x,x01,x02])
    x13 = conv_block(x03, num_filters, 3, act=False)
    sc3 = conv_block(x03, num_filters, 1, act=False)
    x = Activation("relu")(x13 + sc3)
    x = BatchNormalization()(x)
    
    x04 = Concatenate()([x,x01,x02,x03])
    x14 = conv_block(x04, num_filters, 3, act=False)
    sc4 = conv_block(x04, num_filters, 1, act=False)
    x = Activation("relu")(x14 + sc4)
    x = BatchNormalization()(x)
    return x*check

def encoder_block(x, num_filters, length):
    x = multires_block(x, num_filters)
    s = res_path(x, num_filters, length)
    p = MaxPooling2D((2, 2))(x)
    return s, p

def decoder_block(x, skip, num_filters):
    x = Conv2DTranspose(num_filters, 2, strides=2, padding="same")(x)
    x = Concatenate()([x, skip])
    x = multires_block(x, num_filters)
    return x

def build_multiresunet(shape, text):
    """ Input """
    inputs = Input(shape)

    """ Encoder """
    p0 = inputs
    s1, p1 = encoder_block(p0, 32, 4)
    s2, p2 = encoder_block(p1, 64, 4)
    s3, p3 = encoder_block(p2, 128, 4)
    s4, p4 = encoder_block(p3, 256, 4)

    """ Bridge """
    b1 = multires_block(p4, 512)
    dis1, dis2, dis3, dis4 = DistributionModel(text)(b1)

    """ Decoder """
    d1 = decoder_block(b1, s4, 256)
    d1 = d1*dis1
    d2 = decoder_block(d1, s3, 128)
    d2 = d2*dis2
    d3 = decoder_block(d2, s2, 64)
    d3 = d3*dis3
    d4 = decoder_block(d3, s1, 32)
    d4 = d4*dis4
    
    """ Output """
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    """ Model """
    model = Model(inputs, outputs, name="MultiResUNET")

    return model
    
model = build_multiresunet((512, 512, 3), text)
optimizer = AdamW(learning_rate=0.0001)
model.compile(loss=combined_loss, metrics=["accuracy", dice_score, recall, precision, iou], optimizer=optimizer)
model.summary()
