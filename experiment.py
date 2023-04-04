import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

# Dice loss function
def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

# Semantic-guided attention module
def attention_module(inputs):
    attention = layers.Conv2D(1,    (1, 1), activation='sigmoid')(inputs)
    return layers.multiply([inputs, attention])

# Position attention module
def position_attention_module(inputs, inter_channels):
    q = layers.Conv2D(inter_channels, (1, 1), activation='relu')(inputs)
    k = layers.Conv2D(inter_channels, (1, 1), activation='relu')(inputs)
    v = layers.Conv2D(inter_channels, (1, 1), activation='relu')(inputs)

    k = layers.Permute((2, 1, 3))(k)
    matmul_qk = layers.multiply([q, k])
    softmax_qk = layers.Activation('softmax')(matmul_qk)

    matmul_qkv = layers.multiply([softmax_qk, v])
    outputs = layers.add([matmul_qkv, inputs])
    return outputs

# Channel attention module
def channel_attention_module(inputs, reduction_ratio=16):
    channels = inputs.shape[-1]
    avg_pool = layers.GlobalAveragePooling2D()(inputs)
    max_pool = layers.GlobalMaxPooling2D()(inputs)

    dense_1 = layers.Dense(channels // reduction_ratio, activation='relu')
    dense_2 = layers.Dense(channels, activation='sigmoid')

    avg_path =    dense_2(dense_1(avg_pool))
    max_path = dense_2(dense_1(max_pool))
    channel_att = layers.add([avg_path, max_path])

    outputs = layers.multiply([inputs, channel_att])
    return outputs

# Encoder block with attention
def encoder_block(inputs, filters, inter_channels):
    conv = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(inputs)
    conv = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(conv)
    pos_att = position_attention_module(conv, inter_channels)
    channel_att = channel_attention_module(conv)
    outputs = layers.add([pos_att, channel_att])
    return outputs

# Decoder block with attention
def decoder_block(inputs, concat_tensor, filters, inter_channels):
    up = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(inputs)
    up = layers.concatenate([up, concat_tensor])
    conv = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(up)
    conv = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(conv)
    pos_att = position_attention_module(conv, inter_channels)
    channel_att = channel_attention_module(conv)
    outputs = layers.add([pos_att, channel_att])
    return outputs

  
  
  
# UNet with ResNet backbone
def unet_resnet(input_shape):
    inputs = layers.Input(input_shape)
    resnet = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)

    # Encoder
    s1 = resnet.get_layer("input_1").output
    s2 = resnet.get_layer("conv1_relu").output
    s3 = resnet.get_layer("conv2_block3_out").output
    s4 = resnet.get_layer("conv3_block4_out").output

    # Bridge
    b1 = resnet.get_layer("conv4_block6_out").output

    # Encoder blocks with attention
    e1 = encoder_block(s1, 64, 16)
    e2 = encoder_block(s2, 128, 32)
    e3 = encoder_block(s3, 256, 64)
    e4 = encoder_block(s4, 512, 128)

    # Decoder blocks with attention
    d1 = decoder_block(b1, e4, 256, 64)
    d2 = decoder_block(d1, e3, 128, 32)
    d3 = decoder_block(d2, e2, 64, 16)

    # Output
    outputs = layers.Conv2DTranspose(1, (2, 2), strides=(2, 2), padding='same', activation='sigmoid')(d3)

    # Model
    model = models.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss=dice_loss, metrics=['accuracy'])
    return model


# Model instantiation
input_shape = (256, 256, 1)  
unet_model = unet_resnet(input_shape)
unet_model.summary()



