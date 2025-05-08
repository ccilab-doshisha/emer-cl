import typing as t

import tensorflow as tf
from tensorflow.keras.layers import GRU, Bidirectional
from tensorflow.keras.layers import Dense as fc
from tensorflow.keras.layers import Dropout as dropout
from tensorflow.keras.layers import Masking


class MLP(object):
    def __init__(
        self,
        encoder_type: str,  # music or emotion
        emb_dim: int,  # embedding_dim
    ):

        nft = "%s_fc%s"  # name_format (ex. music_fc1)
        act = "softplus"

        # MLP
        self.model = [
            fc(256, activation=act, name=nft % (encoder_type, "1")),
            dropout(0.5),
            fc(512, activation=act, name=nft % (encoder_type, "2")),
            dropout(0.5),
            fc(512, activation=act, name=nft % (encoder_type, "3")),
            dropout(0.5),
            fc(1024, activation=act, name=nft % (encoder_type, "4")),
            dropout(0.5),
            fc(emb_dim, activation=act, name=nft % (encoder_type, "")),
        ]

        self.emb = fc(
            emb_dim, activation="linear", name="%s_emb" % encoder_type
        )
        self.mu = fc(emb_dim, activation="linear", name="%s_mu" % encoder_type)
        self.log_sigma = fc(
            emb_dim, activation="linear", name="%s_sigma" % encoder_type
        )

    # placeholder into Tensor
    def call(self, placeholder: tf.Tensor) -> t.Dict[str, tf.Tensor]:

        x = self.model[0](placeholder)  # first_layer
        for layer in self.model[1:]:  # after second_layer (dropout)
            x = layer(x)

        return {
            "emb": self.emb(x),
            "mu": self.mu(x),
            "log_sigma": self.log_sigma(x),
        }


class BiGRU(object):
    def __init__(
        self,
        encoder_type: str,  # music or emotion
        emb_dim: int,  # embedding_dim
        input_dim: int,  # 128(vggish) or 2(va)
    ):

        nft = "%s_fc%s"  # name_format (ex. music_fc1)
        act = "softplus"
        gru_nft = f"{encoder_type}_bi_gru_512"
        gru_layer = GRU(512, return_sequences=False, name=gru_nft)

        # GRU + MLP
        self.model = [
            Masking(mask_value=1e-5, input_shape=(None, input_dim)),
            Bidirectional(gru_layer, name=gru_nft),
            fc(512, activation=act, name=nft % (encoder_type, "1")),
            dropout(0.5),
            fc(512, activation=act, name=nft % (encoder_type, "2")),
            dropout(0.5),
            fc(512, activation=act, name=nft % (encoder_type, "3")),
            dropout(0.5),
            fc(1024, activation=act, name=nft % (encoder_type, "4")),
            dropout(0.5),
            fc(emb_dim, activation=act, name=nft % (encoder_type, "")),
        ]

        self.emb = fc(
            emb_dim, activation="linear", name="%s_emb" % encoder_type
        )
        self.mu = fc(emb_dim, activation="linear", name="%s_mu" % encoder_type)
        self.log_sigma = fc(
            emb_dim, activation="linear", name="%s_sigma" % encoder_type
        )

    # placeholder into Tensor
    def call(self, placeholder: tf.Tensor) -> t.Dict[str, tf.Tensor]:

        x = self.model[0](placeholder)  # first_layer
        for layer in self.model[1:]:  # after second_layer (gru layer)
            x = layer(x)

        return {
            "emb": self.emb(x),
            "mu": self.mu(x),
            "log_sigma": self.log_sigma(x),
        }


class EMER_CL(object):

    music_encoder: t.Union[MLP, BiGRU]
    emotion_encoder: BiGRU

    def __init__(self, dataset_type: str, emb_dim: int):

        if dataset_type == "DEAM":
            self.music_encoder = MLP("music", emb_dim)
            music_shape = [None, 128]
        else:
            self.music_encoder = BiGRU("music", emb_dim, 128)
            music_shape = [None, None, 128]

        self.emotion_encoder = BiGRU("emotion", emb_dim, 2)

        self.input_music = tf.compat.v1.placeholder(
            tf.float32, shape=music_shape, name="input_music"
        )
        self.input_emotion = tf.compat.v1.placeholder(
            tf.float32, shape=[None, None, 2], name="input_emotion"
        )

    def get_output_tensors(self) -> t.Dict[str, t.Dict[str, tf.Tensor]]:

        output_music_enc = self.music_encoder.call(self.input_music)
        output_emotion_enc = self.emotion_encoder.call(self.input_emotion)

        # embedding, mu, log_sigma
        return {
            "music_enc": output_music_enc,
            "emotion_enc": output_emotion_enc,
        }
