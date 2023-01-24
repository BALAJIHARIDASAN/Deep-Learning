import argparse
import os
import logging
from src.utils.common import read_yaml, create_directories
import tensorflow as tf
import numpy as np

STAGE = "prepare a base model" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path):
    ## read config files
    config = read_yaml(config_path)

    ## get the data
    (X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train_full = X_train_full / 255.0
    X_test = X_test / 255.0
    X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

    # set random seed
    seed = 23
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # define layers
    LAYERS = [ tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(300, kernel_initializer="he_normal"),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dense(100, kernel_initializer="he_normal"),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dense(10, activation="softmax")]

    model = tf.keras.models.Sequential(LAYERS)

    model.compile(loss="sparse_categorical_crossentropy",
              optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3),
              metrics=["accuracy"])

    model.summary()

    # train the model
    history = model.fit(X_train, y_train, epochs=10,
                    validation_data=(X_valid, y_valid), verbose=2) 

    model_dir_path = os.path.join("artifacts", "models")
    create_directories([model_dir_path])
    model_path = os.path.join(model_dir_path, "base_model.h5")
    model.save(model_path)
    logging.info(f"base model is saved at {model_path}")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise 