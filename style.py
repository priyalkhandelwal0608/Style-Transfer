import tensorflow as tf
from tensorflow.keras.applications import vgg19
from tensorflow.keras.models import Model
from utils import gram_matrix

# ----------------------------
# VGG19 Model for Feature Extraction
# ----------------------------
def get_model():
    vgg = vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]

    content_layers = ['block5_conv2']

    outputs = [vgg.get_layer(name).output for name in (style_layers + content_layers)]
    model = Model([vgg.input], outputs)

    return model, style_layers, content_layers


# ----------------------------
# Loss Functions
# ----------------------------
def get_loss(model, loss_weights, init_image, gram_style_features, content_features):
    features = model(init_image)

    style_weight, content_weight = loss_weights

    style_features = features[:len(gram_style_features)]
    content_features_pred = features[len(gram_style_features):]

    # Style loss
    style_score = 0
    for target, comb in zip(gram_style_features, style_features):
        gram_comb = gram_matrix(comb)
        style_score += tf.reduce_mean((gram_comb - target) ** 2)

    # Content loss
    content_score = 0
    for target, comb in zip(content_features, content_features_pred):
        content_score += tf.reduce_mean((comb - target) ** 2)

    style_score *= style_weight / len(gram_style_features)
    content_score *= content_weight / len(content_features)

    loss = style_score + content_score
    return loss, style_score, content_score


@tf.function()
def compute_grads(cfg):
    with tf.GradientTape() as tape:
        all_loss = get_loss(**cfg)
    total_loss = all_loss[0]
    return tape.gradient(total_loss, cfg['init_image']), all_loss


# ----------------------------
# Style Transfer Runner
# ----------------------------
def run_style_transfer(model, content_img, style_img,
                       iterations=300,
                       content_weight=1e4,
                       style_weight=1e-2):

    init_image = tf.Variable(tf.convert_to_tensor(content_img, dtype=tf.float32))
    opt = tf.optimizers.Adam(learning_rate=2.0)

    # Extract features
    style_features = model(style_img)[:5]
    content_features = model(content_img)[5:]

    gram_style_features = [gram_matrix(sf) for sf in style_features]

    loss_weights = (style_weight, content_weight)

    cfg = {
        'model': model,
        'loss_weights': loss_weights,
        'init_image': init_image,
        'gram_style_features': gram_style_features,
        'content_features': content_features
    }

    best_img = None
    best_loss = float('inf')

    for i in range(iterations):
        grads, all_loss = compute_grads(cfg)
        loss, style_score, content_score = all_loss

        opt.apply_gradients([(grads, init_image)])
        init_image.assign(tf.clip_by_value(init_image, -103.939, 255.0 - 103.939))

        if loss < best_loss:
            best_loss = loss
            best_img = init_image.numpy()

        if i % 50 == 0:
            print(f"Iteration {i}: Total={loss:.2f}, Style={style_score:.2f}, Content={content_score:.2f}")

    return best_img